
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings,HuggingFaceInstructEmbeddings
# from langchain.vectorstores import FAISS
from langchain.llms import HuggingFacePipeline, LlamaCpp,CTransformers
from langchain.chains import RetrievalQA
import chainlit as cl
from langchain.vectorstores import Chroma
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import torch
import time
from huggingface_hub import hf_hub_download
from constant import (
    CHROMA_SETTINGS,
    EMBEDDING_MODEL_NAME,
    PERSIST_DIRECTORY,  
)
if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

def get_prompt(instruction, new_system_prompt ):
    SYSTEM_PROMPT = B_SYS + new_system_prompt + E_SYS
    prompt_template =  B_INST + SYSTEM_PROMPT + instruction + E_INST
    return prompt_template.strip()

# PROMPT-1 -------------------------------------------------------------------------------------------------------
# system_prompt = "You are an advanced assistant to help user to find the answer from the provided context.Use the following pieces of information to answer the user's question. If you don't know the answer, just say that you don't know, don't try to make up an answer.\ncontext:{context}"
# instruction = """user_question: {question}"""
# prompt_template=get_prompt(instruction,system_prompt)
# print(prompt_template)

# PROMPT-2 ----------------------------------------------------------------------------------------------------
system_prompt = "Use the following pieces of context about affine leave policy to answer the question at the end. If the answer cannot be found, respond with 'The answer is not available in the given data'.\ncontext:{context}"
instruction = """user_question: {question}"""
prompt_template=get_prompt(instruction,system_prompt)
print(prompt_template)

# PROMPT-3 ------------------------------------------------------------------------------------------------------
# prompt_template = """Use the following pieces of information to answer the user's question.
# If you don't know the answer, just say that you don't know, don't try to make up an answer.
# Context: {context}
# Question: {question}
# Only return the helpful answer below and nothing else.
# Helpful answer:
# """

# PROMPT-4 -------------------------------------------------------------------------------------------------------
# prompt_template="""[INST] <<SYS>>\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Use the following pieces of information to answer the user's question.If you don't know the answer, just say that you don't know, don't try to make up an answer.please return answer only from below context.
#     context:{context} \n<</SYS>>\n\n
#     question:{question}[/INST]
#     """.strip()

#-----------------------------------------------------------------------------------------------------------------
def set_custom_prompt():
    """
    Prompt template for QA retrieval for each vectorstore
    """
    prompt = PromptTemplate(template=prompt_template,
                            input_variables=['context', 'question'])
    return prompt

#Retrieval QA Chain
def retrieval_qa_chain(llm, prompt, db):
    qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                       chain_type='stuff',
                                       retriever=db.as_retriever(search_kwargs={'k': 4}),
                                       return_source_documents=True,
                                       chain_type_kwargs={'prompt': prompt}
                                       )
    return qa_chain

#Loading the model
def load_llm():
    model_id='TheBloke/Llama-2-7B-Chat-GGML'
    model_basename="llama-2-7b-chat.ggmlv3.q4_0.bin"
    max_ctx_size = 2048
    n_gpu_layers = 50
    n_batch = 100
    model_path = hf_hub_download(repo_id=model_id, filename=model_basename)

    llm = LlamaCpp(
    model_path=model_path,
    max_tokens=max_ctx_size,
    n_ctx=max_ctx_size,
    temperature = 0.01,
    n_gpu_layers=n_gpu_layers,
    n_batch=n_batch,
    top_p=0.95,
    repeat_penalty=1.2,
    top_k=50,
    verbose=True)
    return llm

#QA Model Function
def qa_bot():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME,
                                       model_kwargs={'device': device})
    # embeddings = HuggingFaceInstructEmbeddings(model_name=EMBEDDING_MODEL_NAME,
    #                                    model_kwargs={'device': 'cpu'})
    db = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embeddings,
        client_settings=CHROMA_SETTINGS,
    )
    llm = load_llm()
    qa_prompt = set_custom_prompt()
    qa = retrieval_qa_chain(llm, qa_prompt, db)

    return qa

#output function
def final_result(query):
    qa_result = qa_bot()
    response = qa_result({'query': query})
    return response


# while True:
#     query=input("Enter question here:: ")
#     start=time.time()
#     ans=final_result(query)
#     print('-'*50)
#     print(ans["result"])
#     end=time.time()
#     print("TIME",end-start)
#     print('-'*50)

#chainlit code
@cl.on_chat_start
async def start():
    chain = qa_bot()
    msg = cl.Message(content="Starting the bot...")
    await msg.send()
    msg.content = "Hi, Welcome to Affine Bot. What is your query?"
    await msg.update()

    cl.user_session.set("chain", chain)

@cl.on_message
async def main(message):
    chain = cl.user_session.get("chain") 
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"]
    )
    cb.answer_reached = True
    res = await chain.acall(message, callbacks=[cb])
    answer = res["result"]
    # print('*'*70)
    # print("ANSWER::",answer)
    # print('*'*70)
    # sources = res["source_documents"]

    # if sources:
    #     answer += f"\nSources:" + str(sources)
    # else:
    #     answer += "\nNo sources found"

    await cl.Message(content=answer).send()

