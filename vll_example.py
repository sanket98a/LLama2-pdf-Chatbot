import openai
import time
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings,HuggingFaceInstructEmbeddings
# from langchain.vectorstores import FAISS
from langchain.llms import HuggingFacePipeline, LlamaCpp,CTransformers
from langchain.chains import RetrievalQA
import chainlit as cl
from langchain.vectorstores import Chroma
# from langchain.callbacks.manager import CallbackManager
# from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import torch
import time
from constant import (
    CHROMA_SETTINGS,
    EMBEDDING_MODEL_NAME,
    PERSIST_DIRECTORY,  
)
# from langchain.chat_models import ChatOpenAI

from langchain.llms import VLLMOpenAI
import gc
torch.cuda.empty_cache()
gc.collect()

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

system_prompt = "Use the following pieces of context about affine leave policy to answer the question at the end. If the answer cannot be found, respond with 'The answer is not available in the given data'.\n"
instruction = """context:{context} \n user_question: {question}"""
prompt_template=get_prompt(instruction,system_prompt)
print(prompt_template)

def set_custom_prompt():
    """
    Prompt template for QA retrieval for each vectorstore
    """
    prompt = PromptTemplate(template=prompt_template,
                            input_variables=['context', 'question'])
    return prompt

# openai.api_key = "EMPTY"
api_base = "http://3.239.120.9:8000/v1"
start=time.time()
# completion = openai.Completion.create(model="meta-llama/Llama-2-7b-chat-hf",
#                                       prompt="write beutiful poem for me on happiness?",
#                                       temperature=0.95,
#                                       max_tokens=512,
#                                       top_p=1,
#                                       frequency_penalty=0,
#                                       presence_penalty=0)
# end=time.time()

#Retrieval QA Chain
def retrieval_qa_chain(prompt, db):
    llms=VLLMOpenAI(model_name="meta-llama/Llama-2-7b-chat-hf",openai_api_key="EMPTY",openai_api_base=api_base)
    qa_chain = RetrievalQA.from_chain_type(llm=llms,
                                       chain_type='stuff',
                                       retriever=db.as_retriever(search_kwargs={'k': 4}),
                                       return_source_documents=True,
                                       chain_type_kwargs={'prompt': prompt}
                                       )
    return qa_chain

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
    qa_prompt = set_custom_prompt()
    qa = retrieval_qa_chain(qa_prompt, db)

    return qa

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