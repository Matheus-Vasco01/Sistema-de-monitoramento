import os
import sqlite3
import requests
import json
from datetime import datetime, timedelta
from bs4 import BeautifulSoup

# Definindo caminhos de forma isolada para evitar importação circular
SERVICES_DIR = os.path.dirname(os.path.abspath(__file__))
APP_DIR = os.path.dirname(SERVICES_DIR)
DB_PATH = os.path.join(APP_DIR, "detections.db")

G1_AGRO_URL = "https://g1.globo.com/economia/agronegocios/"
CACHE_DURATION_MINUTES = 15

def init_news_db():
    """Inicializa a tabela de notícias no banco de dados SQLite."""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cur = conn.cursor()
            cur.execute("""
                CREATE TABLE IF NOT EXISTS news (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT NOT NULL,
                    url TEXT NOT NULL UNIQUE,
                    summary TEXT,
                    scraped_at TEXT NOT NULL
                )
            """)
            conn.commit()
    except sqlite3.Error as e:
        print(f"[Scraper] Erro ao inicializar tabela de notícias: {e}")

def get_cached_news():
    """Recupera as notícias salvas no banco de dados local."""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            cur.execute("SELECT title, url, summary, scraped_at FROM news ORDER BY id ASC")
            rows = cur.fetchall()
            return [dict(r) for r in rows]
    except sqlite3.Error as e:
        print(f"[Scraper] Erro ao carregar notícias do cache: {e}")
        return []

def save_news_to_cache(news_list):
    """Limpa as notícias antigas e salva as novas no banco de dados."""
    if not news_list:
        return
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cur = conn.cursor()
            # Limpa tabela anterior
            cur.execute("DELETE FROM news")
            
            scraped_at_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            for item in news_list:
                try:
                    cur.execute(
                        "INSERT OR REPLACE INTO news (title, url, summary, scraped_at) VALUES (?, ?, ?, ?)",
                        (item["title"], item["url"], item["summary"], scraped_at_str)
                    )
                except sqlite3.Error:
                    # Ignorar duplicados se houver algum erro de restrição UNIQUE
                    continue
            conn.commit()
    except sqlite3.Error as e:
        print(f"[Scraper] Erro ao salvar notícias no cache SQLite: {e}")

def scrape_g1_agro():
    """Realiza a raspagem de notícias diretamente no G1 Agronegócios."""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/125.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "pt-BR,pt;q=0.9,en;q=0.8"
    }
    
    print(f"[Scraper] Fazendo requisição HTTP para: {G1_AGRO_URL}")
    response = requests.get(G1_AGRO_URL, headers=headers, timeout=10)
    response.raise_for_status()
    
    soup = BeautifulSoup(response.text, "html.parser")
    news_items = []
    
    # G1 usa a classe feed-post-body para agrupar o corpo das notícias do feed
    posts = soup.find_all("div", class_="feed-post-body")
    
    # Se não achar com feed-post-body, tenta buscar por feed-post (estrutura mais genérica)
    if not posts:
        posts = soup.find_all("div", class_="feed-post")
        
    print(f"[Scraper] Encontrados {len(posts)} posts candidatos na página.")
    
    for post in posts[:6]:  # Limitamos às 6 notícias mais recentes
        try:
            # Busca o link e o título do post
            link_tag = post.find("a", class_="feed-post-link")
            if not link_tag:
                # Fallback: pega o primeiro link dentro de um cabeçalho h2 ou h3
                for tag_name in ["h2", "h3", "div"]:
                    parent_tag = post.find(tag_name)
                    if parent_tag:
                        link_tag = parent_tag.find("a")
                        if link_tag:
                            break
            
            if not link_tag:
                continue
                
            title = link_tag.get_text().strip()
            url = link_tag.get("href", "").strip()
            
            if not title or not url:
                continue
                
            # Busca a descrição/resumo da notícia
            summary_tag = post.find("div", class_="feed-post-body-resumo")
            if not summary_tag:
                summary_tag = post.find("div", class_="feed-post-metadata") # Fallback
                
            summary = summary_tag.get_text().strip() if summary_tag else ""
            
            # Limpa espaços excessivos
            summary = " ".join(summary.split())
            if len(summary) > 180:
                summary = summary[:177] + "..."
                
            news_items.append({
                "title": title,
                "url": url,
                "summary": summary if summary else "Clique no link para ler a notícia completa no portal G1."
            })
        except Exception as item_err:
            print(f"[Scraper] Erro ao extrair item de notícia individual: {item_err}")
            continue
            
    return news_items

def fetch_agro_news(force_refresh=False):
    """
    Função principal que gerencia o scraping e o cache de notícias.
    Retorna uma lista de dicionários estruturada em formato JSON.
    """
    init_news_db()
    
    cached = get_cached_news()
    
    # Se houver dados no banco, verificamos o timestamp de coleta do primeiro item
    should_scrape = not cached
    
    if cached and not force_refresh:
        try:
            last_scraped_str = cached[0]["scraped_at"]
            last_scraped = datetime.strptime(last_scraped_str, "%Y-%m-%d %H:%M:%S")
            time_passed = datetime.now() - last_scraped
            
            if time_passed > timedelta(minutes=CACHE_DURATION_MINUTES):
                should_scrape = True
                print(f"[Scraper] Cache expirou ({time_passed.total_seconds() / 60:.1f} minutos). Solicitando novos dados.")
            else:
                print(f"[Scraper] Utilizando dados em cache. Última atualização: {last_scraped_str}")
        except Exception as e:
            # Caso ocorra erro de conversão de data, forçamos o scrape
            print(f"[Scraper] Erro ao ler data de cache, forçando scraping: {e}")
            should_scrape = True
            
    if should_scrape:
        try:
            fresh_news = scrape_g1_agro()
            if fresh_news:
                save_news_to_cache(fresh_news)
                return fresh_news
            else:
                print("[Scraper] Nenhum dado novo retornado pelo scraping. Mantendo cache anterior.")
        except Exception as scrape_err:
            # Fallback seguro: se o site estiver fora do ar ou sem internet, reporta erro e usa cache
            print(f"[Scraper] ERRO ao realizar scraping (Site fora do ar ou sem internet): {scrape_err}")
            print("[Scraper] Ativando Fallback de Segurança: Retornando dados históricos armazenados.")
            
    # Retorna o cache que tivermos (pode ser vazio se for a primeira vez e falhar)
    return get_cached_news()
