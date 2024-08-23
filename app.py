from flask import Flask, jsonify
import os
from langchain import hub
from langchain_openai import ChatOpenAI
from langchain_core.prompts.chat import SystemMessage, HumanMessage
from langchain_core.language_models.chat_models import ChatPromptValue
from typing import List
from dotenv import load_dotenv

app = Flask(__name__)

# Load environment variables from .env file
load_dotenv()

# Set your OpenAI API key
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
# Set your LangSmith API key directly
LANGSMITH_API_KEY = os.getenv('LANGSMITH_API_KEY')
print(OPENAI_API_KEY)

# LangChain and OpenAI setup
obj = hub.pull("wfh/proposal-indexing", api_key=LANGSMITH_API_KEY)
llm = ChatOpenAI(model='gpt-4o-mini', openai_api_key=OPENAI_API_KEY)  
runnable = obj | llm

def split_paragraph_into_topics(paragraph):
    # Create a list of messages
    messages = [
        SystemMessage(content="Split the following paragraph into smaller paragraphs where each new point or topic starts a new paragraph."),
        HumanMessage(content=paragraph)
    ]
    
    # Wrap messages into a ChatPromptValue object
    prompt = ChatPromptValue(messages=messages)
    
    # Generate response
    response = llm.generate([prompt])
    
    # The response should return text split into paragraphs
    split_paragraphs = response.generations[0][0].text.split("\n\n")
    return split_paragraphs


# Utility function to extract text from PDF (you may need to implement this)
def extract_text_from_pdf(filepath):
    # Example implementation using PyMuPDF
    import fitz
    text = ""
    with fitz.open(filepath) as doc:
        for page in doc:
            text += page.get_text()
            print("text : ",text)
    return text

@app.route('/upload', methods=['GET'])
def upload_pdf():
    # Assuming the PDF file is in the local directory
    file = "iea_1872.pdf"

    if not file.endswith('.pdf'):
        return jsonify({"error": "Invalid file type"}), 400

    filepath = os.path.join(os.getcwd(), file)

    # Extract text from the PDF
    text = extract_text_from_pdf(filepath)

    # Assuming the text is a single large paragraph
    # Split the text into paragraphs by topics
    split_paragraphs = split_paragraph_into_topics(text)

    return jsonify({"result": split_paragraphs}), 200


if __name__ == '__main__':
    app.run(debug=True)
