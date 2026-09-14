# Como publicar este projeto de graça (sem rodar nada localmente)

O jeito mais simples e gratuito é o **Streamlit Community Cloud**. Não pede
cartão de crédito, só uma conta GitHub (também grátis). Depois de publicado,
você recebe um link público (tipo `https://seu-app.streamlit.app`) que
qualquer pessoa acessa, e o site fica com o painel interativo pronto
(`app.py`) — frequências, testes, Monte Carlo, modelos, gerador e
fechamento, tudo clicável no navegador.

## Passo a passo (uns 5 minutos)

### 1. Criar uma conta no GitHub (se ainda não tiver)
Vá em [github.com](https://github.com) → "Sign up". É grátis.

### 2. Criar um repositório novo
- Clique em "New repository".
- Dê um nome (ex: `mega-sena-analise`).
- Marque como **Public** (o plano grátis do Streamlit Cloud exige repositório público).
- Não precisa marcar nada de README/gitignore — vamos subir os arquivos prontos.

### 3. Subir os arquivos do projeto
Duas formas, escolha a mais fácil pra você:

**Opção A — pela interface do GitHub (sem terminal):**
Na página do repositório recém-criado, clique em "uploading an existing
file" e arraste TODA a pasta `mega_sena/` (todos os arquivos e subpastas:
`app.py`, `src/`, `dados/`, `requirements.txt`, `.streamlit/`, etc.).
O GitHub aceita arrastar pastas inteiras direto do navegador.

**Opção B — pelo terminal (se você usa Git no dia a dia):**
```bash
cd mega_sena
git init
git add .
git commit -m "primeira versão"
git branch -M main
git remote add origin https://github.com/SEU-USUARIO/mega-sena-analise.git
git push -u origin main
```

### 4. Publicar no Streamlit Community Cloud
- Vá em [share.streamlit.io](https://share.streamlit.io).
- Clique em "Sign in with GitHub" e autorize.
- Clique em "New app".
- Escolha o repositório que você acabou de criar.
- Em "Main file path", digite: `app.py`
- Clique em "Deploy".

Em 1-2 minutos o app estará no ar, com uma URL pública que você pode
compartilhar com qualquer pessoa.

## Atualizando o app depois de publicado

Toda vez que você subir uma mudança para o GitHub (novo `git push`, ou
editar um arquivo direto pela interface do GitHub), o Streamlit Cloud
detecta e atualiza o app publicado automaticamente — não precisa
republicar manualmente.

## Se quiser trocar a base de dados depois de publicado

O app já tem, na barra lateral, a opção "Enviar meu próprio CSV" — dá pra
testar com uma base atualizada sem mexer no código nem republicar nada,
direto pela interface web.

## Alternativa (se o Streamlit Cloud não funcionar pra você)

**Hugging Face Spaces** ([huggingface.co/spaces](https://huggingface.co/spaces))
também hospeda apps Streamlit de graça e funciona de forma parecida: crie
uma conta, clique em "Create new Space", escolha "Streamlit" como SDK, e
suba os mesmos arquivos (pode ser feito direto pela interface web deles,
sem git). O arquivo principal também deve se chamar `app.py`.

## Limitações do plano gratuito (para não ter surpresa)

- O app "dorme" depois de um tempo sem uso e demora alguns segundos para
  "acordar" na próxima visita — normal em qualquer plano grátis.
- Recursos de CPU/memória são limitados, por isso a simulação Monte Carlo
  no app web tem um limite de repetições ajustável por slider (até 5.000)
  em vez das 20.000 usadas no relatório gerado localmente — para rodar
  mais, use `python3 executar_pipeline.py` na sua máquina.
