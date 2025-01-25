import argparse
# from langchain.vectorsotres.chroma import Chroma
# from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
# from langchain_community.llms.ollama import Ollama
from langchain_ollama import OllamaLLM
from embed import embed

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("question", type=str)
    args = parser.parse_args()
    question = args.question
    ask(question)

def ask(question: str):

    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embed())

    # cos sim
    res = db.similarity_search_with_score(question, k=5)

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in res])
    # print(context_text) --> to get sources chunks
    template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = template.format(context=context_text, question=question)

    model = OllamaLLM(model="llama3.1")
    response = model.invoke(prompt)

    sources = [doc.metadata.get("id", None) for doc, _score in res]

    formatted_response = f"Response: {response}\nSources: {sources}"
    print(formatted_response)
    return response

if __name__ == "__main__":
    main()
