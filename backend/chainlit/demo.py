import chainlit as cl
import anthropic
# from langchain.schema.runnable.config import RunnableConfig
# cb = cl.LangchainCallbackHandler(stream_final_answer=True)
# config = RunnableConfig(callbacks=[cb])
# result = agent.invoke(input, config=config)


def answer(q):
    import urllib.parse
    import os
    from dotenv import load_dotenv
    from pymongo import MongoClient

    username = urllib.parse.quote_plus('anantbhaai_1')
    password = urllib.parse.quote_plus('anantbhai@123')

    from langchain.embeddings import OpenAIEmbeddings
    from langchain.vectorstores import MongoDBAtlasVectorSearch
    from langchain.document_loaders import PyPDFDirectoryLoader
    # from langchain.embeddings import OpenAIEmbeddings
    # from langchain.vectorstores import MongoDBAtlasVectorSearch

    load_dotenv(override=True)

    MONGO_URI = "mongodb+srv://%s:%s@cluster0.pjuurgm.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0" % (
        username, password)

    DB_NAME = "bha_ai-test"
    COLLECTION_NAME = "test"
    ATLAS_VECTOR_SEARCH_INDEX_NAME = "vector_index"

    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]

    collection_list = db.list_collection_names()

    # Check if the collection exists, if not, create it
    if COLLECTION_NAME not in collection_list:
        db.create_collection(COLLECTION_NAME)

    MONGODB_COLLECTION = db[COLLECTION_NAME]

    loader = PyPDFDirectoryLoader("pdfs")
    data = loader.load()
    print(data)

    from langchain.text_splitter import RecursiveCharacterTextSplitter

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000,
                                                   chunk_overlap=0)
    docs = text_splitter.split_documents(data)

    openai_api_key = os.environ.get('OPENAI_API_KEY')

    # insert the documents in MongoDB Atlas Vector Search
    x = MongoDBAtlasVectorSearch.from_documents(
        documents=docs,
        embedding=OpenAIEmbeddings(openai_api_key=openai_api_key,
                                   disallowed_special=()),
        collection=MONGODB_COLLECTION,
        index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME)

    vector_search = MongoDBAtlasVectorSearch.from_connection_string(
        MONGO_URI,
        DB_NAME + "." + COLLECTION_NAME,
        OpenAIEmbeddings(openai_api_key=openai_api_key, disallowed_special=()),
        index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME)

    results = vector_search.similarity_search(
        query, k=3)  # Retrieve top 5 similar documents

    # client = anthropic.Client(api_key=userdata.get("ANTHROPIC_API_KEY"))
    client = anthropic.Client(
        api_key= os.environ.get('ANTHROPIC_API_KEY')
    )

    def query_claude(query):
        # Perform a similarity search query
        results = vector_search.similarity_search(
            query, k=3)  # Retrieve top 3 similar documents

        # Concatenate the context from search results
        context = ""
        for result in results:
            context += result.page_content
    # 2. Identify questions related to credit cards were  asked.

        response = client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=1024,
            # system = "You are a knowledgeable assistant focused on providing accurate and relevant information about credit cards. Use the context provided to give detailed answers specifically related to credit cards. If the context includes numerical data, make sure to calculate and present the information accurately. Avoid giving general responses or straying from the credit card topic.",
            # Prepare the prompt for Claude API
            system=
            """I am sharing multiple call document for analysis in the context. You need to analyse the documents pertaining to each call separately and update the output section with your analysis. Instructions for Analysis are:
            1. Identify how many credit card-related questions are asked.
            2. only answer the correct answer.
            3. always do the correct calculation""",
            messages=[{
                "role":
                "user",
                "content":
                "Answer this user query: " + query +
                " with the following context: " + context
            }])
        print(response.content[0].text)

    # Example usage
    # query = "tell me all about the intersest rate in credit card"

    query = "tell me all about the intersest rate in credit card"

    result = query_claude(q)
    return result


def query_model(query):
    import os
    client = anthropic.Client(
        api_key= os.environ.get('ANTHROPIC_API_KEY')
    )

    # system = "You are a knowledgeable assistant focused on providing accurate and relevant information about credit cards. Use the context provided to give detailed answers specifically related to credit cards. If the context includes numerical data, make sure to calculate and present the information accurately. Avoid giving general responses or straying from the credit card topic.",
    # Prepare the prompt for Claude API
    # system=
    # """I am sharing multiple call document for analysis in the context. You need to analyse the documents pertaining to each call separately and update the output section with your analysis. Instructions for Analysis are:
    # 1. Identify how many credit card-related questions are asked.
    # 2. only answer the correct answer.
    # 3. always do the correct calculation""",
    response = client.messages.create(
        model="claude-3-sonnet-20240229",
        max_tokens=1024,
        messages=[{
            "role": "user",
            "content": "Answer this user query: " + query
        }])
    return (response.content[0].text)

    # Example usage
    # query = "tell me all about the intersest rate in credit card"

    # query = "tell me all about the intersest rate in credit card"

    # result = query_claude(q)
    # return result


@cl.step(type="tool")
async def tool(query):
    # Fake tool
    await cl.sleep(2)
    return query_model(query)


@cl.on_message  # this function will be called every time a user inputs a message in the UI
async def main(message: cl.Message):
    """
    This function is called every time a user inputs a message in the UI.
    It sends back an intermediate response from the tool, followed by the final answer.

    Args:
        message: The user's message.

    Returns:
        None.
    """
    final_answer = await cl.Message(content="").send()

    # Call the tool
    answer = await tool(message.content)
    final_answer.content = await tool(answer)

    await final_answer.update()
