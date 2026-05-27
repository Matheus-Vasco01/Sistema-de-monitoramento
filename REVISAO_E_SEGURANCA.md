# 📝 Relatório de Revisão de Arquitetura, Segurança e Melhoria de Código

Este documento apresenta a análise crítica de arquitetura, riscos de segurança, a revisão de trechos de código e a justificativa técnica da camada de web scraping para o projeto **AgroVision AI**.

---

## 🏛️ Parte 1 — Revisão da Arquitetura

O sistema possui uma divisão conceitual entre as camadas, mas elas estavam fisicamente acopladas no arquivo `app.py`. Abaixo estão as respostas para os questionamentos de arquitetura:

1.  **A interface está apenas exibindo dados ou também possui regra de negócio indevida?**
    *   **Resposta**: A interface (`index.html`) atua principalmente como camada de apresentação. Contudo, ela possui lógica de streaming em JavaScript diretamente embutida para tratar as chunks do chat (`/chat/stream`). Não há regras de negócio críticas de agronegócio ou monitoramento no frontend; as regras residem no backend.
2.  **O backend concentra a lógica principal do sistema?**
    *   **Resposta**: Sim, o FastAPI (`app.py`) centraliza toda a lógica de negócio do sistema: gerencia as rotas HTTP/Stream, inicia e gerencia a thread de captura e inferência da câmera, interage com o banco de dados e envia dados contextualizados para a API do Ollama.
3.  **O acesso ao banco está isolado em uma camada própria ou aparece espalhado pelo código?**
    *   **Resposta**: O acesso está parcialmente isolado em funções auxiliares (`init_db`, `save_event`, `list_events`) no próprio `app.py`. No entanto, essas funções residem no mesmo arquivo monolítico que contém as rotas web e a inferência de IA. O ideal é mover a persistência para um módulo específico (ex: `database.py` ou `repositories/`).
4.  **A chamada ao modelo de IA/YOLO está separada da regra de negócio?**
    *   **Resposta**: Não. A inferência do modelo YOLO (`model(frame)`) ocorre acoplada dentro do loop infinito da thread de captura (`process_stream`), que decide se grava arquivos no disco e persiste no banco de dados. Essa mistura dificulta a manutenção e testes isolados.
5.  **A nova camada de scraping será implementada como serviço separado ou ficará misturada em rotas, telas ou controllers?**
    *   **Resposta**: A nova camada de scraping foi isolada em um serviço separado sob o arquivo [scraping_service.py](file:///c:/Users/Mathe/OneDrive/Documents/Materia_IA/Sistema-de-monitoramento/agrovision_ia/app/services/scraping_service.py). O backend (`app.py`) apenas invoca este serviço por meio de uma rota de API dedicada (`/api/news`), mantendo a separação física de responsabilidades.

---

## 🔒 Parte 2 — Revisão de Segurança

Identificamos os seguintes riscos e oportunidades de melhoria de segurança no projeto atual:

1.  **Chaves, Senhas ou Tokens no Código**: Não há credenciais de produção diretamente expostas em formato plaintext, mas chaves de câmera RTSP e IPs de infraestrutura estão expostos em variáveis hardcoded ou comentários. O recomendado é usar arquivos `.env` para isolar essas variáveis do controle de versão.
2.  **Rotas da API Sem Validação**: Todas as rotas do FastAPI (`/chat`, `/video_feed`, etc.) estão abertas a qualquer dispositivo na rede local. Isso permite acesso não autorizado às câmeras de monitoramento rural e abuso de requisições de IA local (levando a ataques de Negação de Serviço por consumo de CPU/GPU).
3.  **Validação de Dados Enviados**: A validação inicial ocorre via Pydantic (`ChatRequest`), mas não há qualquer sanitização contra ataques de *Prompt Injection* antes de enviar os dados ao Ollama, o que poderia forçar a IA a se comportar de forma não desejada.
4.  **Injeção de SQL e Exposição de Dados**:
    *   O SQL Injection é mitigado com sucesso pela parametrização das queries (`?`).
    *   Há risco de exposição de dados porque o tráfego MJPEG (`/video_feed`) é transmitido por protocolo HTTP sem criptografia SSL/TLS e sem qualquer controle de cookies/tokens de sessão.
5.  **Tratamento Seguro de Erros**: O tratamento de exceções nas rotas do chat expunha a mensagem técnica interna do Python (`str(e)`) ao usuário final em formato JSON, revelando detalhes de infraestrutura (como conexões de portas locais ou nomes de arquivos).
6.  **Fontes não Confiáveis**: Ao consumir streams externos via `yt-dlp` ou ao raspar páginas externas, o sistema fica vulnerável a dados maliciosos. No Scraping, é vital tratar o texto coletado para evitar ataques de *XSS Armazenado* na renderização do dashboard.

---

## 🛠️ Parte 3 — Melhoria do Código Gerado com IA

Selecionamos três trechos de código que possuem problemas estruturais ou de performance típicos de código gerado por IA para realizar uma revisão crítica e aplicar correções.

### 🔴 Trecho 1: Operações de Banco de Dados sem Contexto e Tratamento de Erros
*   **Arquivo**: [app.py](file:///c:/Users/Mathe/OneDrive/Documents/Materia_IA/Sistema-de-monitoramento/agrovision_ia/app/app.py) (`save_event`)

#### Código Original:
```python
def save_event(event_id, label, confidence, image_path):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("INSERT INTO events VALUES (?, ?, ?, ?, ?)", 
               (event_id, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), label, confidence, image_path))
    conn.commit()
    conn.close()
```

#### Problema Encontrado:
O código abre e fecha a conexão manualmente sem usar blocos de tratamento de exceção (`try/except/finally`) ou gerenciadores de contexto (`with`). Se ocorrer um erro durante a execução da query (por exemplo, banco bloqueado ou chave primária duplicada), a instrução `conn.close()` nunca será chamada, causando vazamento de conexões abertas e posterior travamento (lock) do SQLite.

#### Nova Versão Proposta:
```python
def save_event(event_id, label, confidence, image_path):
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO events VALUES (?, ?, ?, ?, ?)", 
                (event_id, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), label, confidence, image_path)
            )
            conn.commit()
    except sqlite3.Error as e:
        print(f"Erro ao salvar evento {event_id} no SQLite: {e}")
```

#### Por que é melhor?
O uso de `with` garante que a conexão seja finalizada corretamente mesmo se uma exceção ocorrer. O bloco `try/except` captura erros de banco sem interromper a execução do fluxo da thread de vídeo e evita o travamento das conexões.

---

### 🔴 Trecho 2: Chamada Síncrona Bloqueante no FastAPI Event Loop
*   **Arquivo**: [app.py](file:///c:/Users/Mathe/OneDrive/Documents/Materia_IA/Sistema-de-monitoramento/agrovision_ia/app/app.py) (`chat`)

#### Código Original:
```python
@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    try:
        answer, messages, r_time = ask_ollama(req.message, req.history, req.model or MODEL_NAME)
        new_history = req.history + [Message(role="assistant", content=answer)]
        return ChatResponse(answer=answer, history=new_history, response_time=r_time)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
```

#### Problema Encontrado:
A rota é definida como assíncrona (`async def chat`), mas ela chama `ask_ollama`, que executa uma chamada de rede puramente síncrona/bloqueante usando a biblioteca `requests.post`. Como o FastAPI gerencia rotas `async` em uma única thread (Event Loop), essa requisição de rede bloqueia o loop inteiro para todos os usuários conectados enquanto o Ollama processa a resposta da IA (o que pode levar segundos).

#### Nova Versão Proposta:
```python
from fastapi.concurrency import run_in_threadpool

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    try:
        # Executa a função síncrona ask_ollama no pool de threads do FastAPI sem bloquear o loop
        answer, messages, r_time = await run_in_threadpool(
            ask_ollama, req.message, req.history, req.model or MODEL_NAME
        )
        new_history = req.history + [Message(role="assistant", content=answer)]
        return ChatResponse(answer=answer, history=new_history, response_time=r_time)
    except Exception as e:
        print(f"Erro na rota de chat com IA: {e}")
        return JSONResponse(status_code=500, content={"error": "Erro ao processar sua pergunta com a IA local."})
```

#### Por que é melhor?
O uso de `run_in_threadpool` desvia a requisição pesada/bloqueante do Ollama para uma thread separada do pool do FastAPI, permitindo que o servidor gerencie dezenas de conexões ao mesmo tempo no event loop principal. O tratamento de erro foi melhorado para esconder stack traces técnicas e conexões vazadas.

---

### 🔴 Trecho 3: Espera Concorrente Ineficiente Mantendo o Lock Ativo (Deadlock Parcial)
*   **Arquivo**: [app.py](file:///c:/Users/Mathe/OneDrive/Documents/Materia_IA/Sistema-de-monitoramento/agrovision_ia/app/app.py) (`generate_mjpeg_stream`)

#### Código Original:
```python
def generate_mjpeg_stream():
    while True:
        with last_frame_lock:
            if last_frame is None:
                time.sleep(0.1)
                continue
            _, buffer = cv2.imencode(".jpg", last_frame)
            frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.05)
```

#### Problema Encontrado:
Dentro da cláusula `with last_frame_lock:`, se o frame ainda for `None`, o código executa `time.sleep(0.1)` mantendo o lock bloqueado para a thread inteira. Como a thread de captura (`process_stream`) precisa adquirir o mesmo `last_frame_lock` para salvar o frame atualizado, a thread de processamento ficará bloqueada esperando a liberação do lock. Isso gera um gargalo de concorrência massivo, reduzindo severamente o framerate do vídeo.

#### Nova Versão Proposta:
```python
def generate_mjpeg_stream():
    while True:
        frame_to_encode = None
        with last_frame_lock:
            if last_frame is not None:
                frame_to_encode = last_frame.copy()
        
        if frame_to_encode is None:
            time.sleep(0.1) # Aguarda fora do bloco lock, permitindo que a thread produtora escreva o frame
            continue
            
        _, buffer = cv2.imencode(".jpg", frame_to_encode)
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.05)
```

#### Por que é melhor?
O lock é mantido aberto pelo menor tempo possível (apenas para copiar a referência do frame na memória). Qualquer tempo de espera de rede ou E/S (`time.sleep`) é feito fora do lock, o que elimina qualquer travamento de concorrência entre a thread produtora (que captura imagens da câmera) e a consumidora (que renderiza a tela).

---

## 🌾 Parte 4 — Implementação de uma Camada de Web Scraping

Para enriquecer o monitoramento rural efetuado pela IA (YOLO), implementamos uma camada de web scraping dedicada a coletar e centralizar notícias do agronegócio nacional.

### 🔍 Justificativa e Finalidade do Dado Coletado
O monitoramento de câmeras foca em eventos físicos locais (passagem de veículos, movimentações suspeitas, segurança patrimonial). No entanto, o gestor do agronegócio necessita de visões macroeconômicas e climáticas para planejar suas operações diárias (logística de transporte de safra, períodos de colheita, cotação de insumos).
*   Ao disponibilizar notícias em tempo real do portal **G1 Agronegócios** no mesmo painel de câmeras, o sistema unifica informações operacionais locais (segurança da fazenda) com dados setoriais externos, otimizando a tomada de decisão do produtor de forma prática e imediata.

### 📐 Estrutura Técnica e Boas Práticas Adotadas
Conforme os requisitos técnicos exigidos, a solução contempla:

1.  **Módulo Separado de Scraping**: O scraping está isolado no arquivo [scraping_service.py](file:///c:/Users/Mathe/OneDrive/Documents/Materia_IA/Sistema-de-monitoramento/agrovision_ia/app/services/scraping_service.py), mantendo a arquitetura limpa e independente de rotas e controllers.
2.  **Uso de Fonte Pública e Gratuita**: Utilizamos o portal público [G1 Agro](https://g1.globo.com/economia/agronegocios/) como fonte de notícias.
3.  **Controle de Requisições por Cache local**: Implementamos um cache de banco de dados SQLite (tabela `news` no `detections.db`) com duração padrão de **15 minutos**. Desta forma, o script só realiza uma nova requisição HTTP externa se o cache expirar, economizando tráfego de rede e prevenindo o bloqueio por excesso de chamadas.
4.  **Tratamento Robustos de Erro (Fallback de Segurança)**: Caso a conexão HTTP falhe (portal fora do ar ou perda de conexão local), o sistema captura a exceção de rede silenciosamente e ativa o fallback de segurança, retornando as notícias históricas presentes no banco de dados local.
5.  **Organização Estruturada**: Os dados coletados são mapeados para listas de dicionários contendo `title`, `url`, `summary` e `scraped_at`, prontamente representáveis como JSON para fácil transmissão via API.
6.  **Integração no Sistema**: O backend expõe a rota `/api/news` que consome o serviço. No frontend, o componente de notícias é carregado dinamicamente via AJAX no carregamento inicial da página e pode ser atualizado manualmente com um clique sem recarregar o dashboard.
