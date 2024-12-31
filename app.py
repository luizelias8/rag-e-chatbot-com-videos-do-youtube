import os
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi

# Carrega as variáveis de ambiente
load_dotenv()

# Inicializar o modelo de chat
chat = ChatGroq(
    api_key=os.getenv('GROQ_API_KEY'), # Chave de API
    model='llama-3.3-70b-versatile', # Modelo LLM a ser usado
    temperature=0.2, # Baixa temperatura para respostas mais precisas
    max_tokens=500 # Limite de tokens na resposta
)

def obter_transcricao_youtube(links):
    """Obtém a transcrição de vídeos do YouTube a partir de uma lista de links."""
    # Inicializa a variável que irá armazenar as transcrições concatenadas
    documento = ''
    for link in links:
        try:
            # Extrai o ID do vídeo a partir do link fornecido
            video_id = link.split('v=')[-1].split('&')[0]
            # Obtém a transcrição para os idiomas português e inglês
            transcricao = YouTubeTranscriptApi.get_transcript(video_id, languages=['pt', 'en'])
            for trecho in transcricao:
                # Concatena os textos da transcrição em um único documento
                documento += trecho['text'] + ' '
        except Exception as e:
            # Exibe um aviso se ocorrer um erro ao processar o vídeo
            st.warning(f'Erro ao processar o vídeo: {link}. Erro: {e}')
    return documento

def obter_base_vetores_dos_textos(texto):
    """Divide o texto em pedaços e cria uma base vetorial."""

    # Configura o divisor de texto em pedaços
    divisor_texto = CharacterTextSplitter(
        separator=' ', # Define o espaço como separador
        chunk_size=500, # Define o tamanho de cada pedaço de texto
        chunk_overlap=200, # Define a sobreposição entre os pedaços
        length_function=len # Usa o comprimento do texto para controle de tamanho
    )

    # Divide o texto do documento em pedaços
    pedacos_documento = divisor_texto.split_text(texto)

    # Configura o modelo de embeddings para gerar representações vetoriais
    modelo_embeddings = HuggingFaceEmbeddings(
        model_name='sentence-transformers/all-MiniLM-L6-v2', # Nome do modelo de embeddings
        model_kwargs={'device': 'cpu'}, # Define o uso do CPU para processamento
        encode_kwargs={'normalize_embeddings': False} # Não aplica normalização aos embeddings
    )

    # Cria uma base vetorial persistente usando os textos em pedaços
    base_vetores = FAISS.from_texts(pedacos_documento, modelo_embeddings)
    return base_vetores

def montar_prompt(fragmentos, pergunta):
    """Monta manualmente o prompt com os fragmentos e o histórico de conversa."""

    template = """
    Use os trechos fornecidos para responder à pergunta do usuário de forma clara e concisa.
    Se necessário, complemente a resposta utilizando o histórico do chat.
    Se não souber a resposta com base nos trechos fornecidos e no histórico do chat, diga que não sabe, sem tentar adivinhar ou inventar informações.
    Se possível, seja direto e objetivo ao responder.

    ### Trechos:
    {fragmentos}

    ### Pergunta:
    {pergunta}
    """

    # Juntar todos os fragmentos em um único texto
    contexto = '\n'.join([f'{indice}. {fragmento.page_content}\n' for indice, fragmento in enumerate(fragmentos,1)])

    # Criar e formatar o prompt
    prompt = template.format(fragmentos=contexto, pergunta=pergunta)

    return prompt

def main():
    """Função principal para configurar e executar a interface da aplicação Streamlit."""
    # Inicializa o histórico de chat na sessão, se ainda não existir
    if 'historico_chat' not in st.session_state:
        st.session_state.historico_chat = []
    # Inicializa a base de vetores na sessão, se ainda não existir
    if 'base_vetores' not in st.session_state:
        st.session_state.base_vetores = None

    # Configura o título e o ícone da página
    st.set_page_config(page_title='Chat com vídeos do YouTube', page_icon='🤖')
    st.title('Chat com vídeos do YouTube')

    # Configura a barra lateral para upload de arquivo
    with st.sidebar:
        # Cabeçalho das configurações
        st.header('🔗 Links de Vídeos')
        # Permite colar URLs de vídeos do YouTube
        links_videos = st.text_area(
            'Cole aqui os links dos vídeos do YouTube',
            help='Apenas vídeos públicos com transcrições ativadas são suportados.'
        )
        if st.button('Processar Vídeos', use_container_width=True):
            # Mostra spinner durante processamento
            with st.spinner('Processando vídeos...'):
                # Inicializa o histórico de chat com a primeira mensagem do bot
                st.session_state.historico_chat.append(AIMessage(content='Olá, sou um bot. Como posso ajudar?'))
                # Obtém o texto consolidado das transcrições
                texto_completo = obter_transcricao_youtube(links_videos.splitlines())
                # Gera a base vetorial
                st.session_state.base_vetores = obter_base_vetores_dos_textos(texto_completo)
            st.success('Vídeos processados com sucesso!')

    # Se a base vetorial existir, permite interação no chat
    if st.session_state.base_vetores is not None:
        # Captura a entrada do usuário no chat
        pergunta = st.chat_input('Digite sua mensagem aqui...')
        # Processa a mensagem do usuário e gera resposta
        if pergunta is not None and pergunta != '':
            # Recuperar documentos relevantes com base na pergunta usando o banco vetorial
            documentos_relevantes = st.session_state.base_vetores.similarity_search(pergunta, k=3)

            # Montar o prompt com os fragmentos
            prompt = montar_prompt(documentos_relevantes, pergunta)

            # Adiciona o prompt com os trechos e a pergunta ao histórico
            st.session_state.historico_chat.append(HumanMessage(content=prompt))

            # Exibir um spinner enquanto o modelo gera a resposta
            with st.spinner('Gerando resposta...'):
                resposta = chat.invoke(st.session_state.historico_chat) # Obter a resposta do modelo

            # Limpa o histórico antes de adicionar a resposta, removendo o prompt montado
            st.session_state.historico_chat.pop() # Remove a última, que seria o prompt montado
            st.session_state.historico_chat.append(HumanMessage(content=pergunta)) # Adiciona apenas a pergunta ao histórico
            st.session_state.historico_chat.append(AIMessage(content=resposta.content)) # Adicionar a resposta do modelo ao histórico de mensagens

        # Exibe o histórico do chat na interface
        for mensagem in st.session_state.historico_chat:
            if isinstance(mensagem, AIMessage): # Mensagem do chatbot
                with st.chat_message('ai'):
                    st.write(mensagem.content)
            elif isinstance(mensagem, HumanMessage): # Mensagem do usuário
                with st.chat_message('human'):
                    st.write(mensagem.content)

# Executa a aplicação se o script for chamado diretamente
if __name__ == '__main__':
    main()