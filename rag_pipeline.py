import os
import getpass
import re
import json
from dotenv import load_dotenv
from typing import Any, Dict, List

# ------------------------
# Groq / LangChain Imports
# ------------------------
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.output_parsers import PydanticOutputParser

# ------------------------
# Additional LangChain Tools
# ------------------------
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

# ------------------------
# LLamaIndex for Data Reading
# ------------------------
from llama_index.core import SimpleDirectoryReader

# ------------------------
# Pydantic for Structured Output
# ------------------------
from pydantic import BaseModel, Field, ValidationError


class RAGResponse(BaseModel):
    """
    Defines the structured JSON response for a RAG pipeline query.
    """
    answer: str = Field(description="Comprehensive answer to the query.")
    source_documents: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of source documents used to generate the answer."
    )
    confidence_score: float = Field(
        default=0.0,
        ge=0,
        le=1,
        description="Confidence of the answer (scale: 0 to 1)."
    )


class RAGPipeline:
    """
    A class that encapsulates a Retrieval-Augmented Generation (RAG) pipeline
    using Chroma for vector storage and the Groq model API for answering queries.
    """

    def __init__(
        self,
        data_dir: str,
        model_name: str = "llama3-8b-8192",
        huggingface_embed_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        groq_api_key: str = None,
    ):
        """
        Initializes the RAG pipeline by:
          1. Setting up the Groq API key (if not already set in the environment).
          2. Instantiating the ChatGroq model.
          3. Loading documents from the specified directory.
          4. Splitting and embedding the texts.
          5. Storing them in Chroma vector storage.

        :param data_dir: Directory containing your documents (PDFs, etc.).
        :param model_name: Name of the Groq model to use.
        :param huggingface_embed_model: Name of the Hugging Face embedding model.
        :param groq_api_key: Optional explicit Groq API key; if not supplied, user is prompted.
        """

        # ---- Set Groq API Key ----
        
        # Load variables from .env into os.environ (if .env exists)
        load_dotenv()

        # Check if .env provided a GROQ_API_KEY
        groq_api_key_in_env = os.environ.get("GROQ_API_KEY")

        if groq_api_key and not groq_api_key_in_env:
            # If the caller explicitly passed a key, but none from .env,
            # set the environment to the provided key.
            os.environ["GROQ_API_KEY"] = groq_api_key
        elif not groq_api_key_in_env:
            # No key from argument or .env, prompt the user as a fallback
            os.environ["GROQ_API_KEY"] = getpass.getpass("Enter API key for Groq: ")

        # ---- Initialize ChatGroq Model ----
        self.model = ChatGroq(model=model_name)

        # ---- Load Documents ----
        self.documents = SimpleDirectoryReader(data_dir).load_data()

        # ---- Create Embeddings ----
        self.embeddings_model = HuggingFaceEmbeddings(model_name=huggingface_embed_model)

        # ---- Split Documents into Chunks ----
        self.split_texts = self._split_documents(self.documents)

        # ---- Store in Chroma ----
        self.vectorstore = self._store_in_chroma(self.split_texts, self.embeddings_model)

        # ---- Create Retriever ----
        self.retriever = self.vectorstore.as_retriever()

        # ---- Compile Prompt for Basic QA ----
        self.basic_prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template=(
                "You are a helpful assistant. Use the following pieces of context "
                "to answer the question at the end.\nIf you don't know the answer, just say "
                "you don't know. DO NOT make up an answer.\n\nContext:\n{context}\n\n"
                "Question:\n{question}\n\nAnswer:\n"
            ),
        )

        # ---- Setup Enhanced Prompt for Structured Output ----
        self.output_parser = PydanticOutputParser(pydantic_object=RAGResponse)
        format_instructions = self.output_parser.get_format_instructions()
        enhanced_template = (
            "You are a precise and helpful research assistant.\n"
            "Analyze the provided context carefully and answer the question with the following guidelines:\n"
            "1. Provide a comprehensive and accurate answer based strictly on the given context.\n"
            "2. If the answer cannot be definitively found in the context, acknowledge this explicitly.\n"
            "3. Assess and report your confidence in the answer.\n"
            "4. Cite the specific sources used for your answer.\n\n"
            f"{format_instructions}\n\n"
            "Context:\n{{context}}\n\n"
            "Question: {{question}}\n\n"
            'Provide your response DIRECTLY as a JSON object matching the specified format.\n'
            'Do NOT nest the response inside another key like "example".'
        )

        self.enhanced_prompt_template = PromptTemplate(
            template=enhanced_template,
            input_variables=["context", "question"]
        )

    def basic_query(self, query: str, top_k: int = 3) -> str:
        """
        Answers a query using the basic prompt template and returns the raw text response.

        :param query: The user’s question.
        :param top_k: Number of top similar documents to retrieve.
        :return: String response from the model.
        """
        docs = self.retriever.get_relevant_documents(query)[:top_k]
        context = "\n".join([doc.page_content for doc in docs])
        prompt_text = self.basic_prompt_template.format(context=context, question=query)

        response = self.model([HumanMessage(content=prompt_text)])
        return response.content

    def enhanced_query(self, query: str, top_k: int = 3) -> RAGResponse:
        """
        Answers a query using a structured output format (JSON) with confidence and source docs.

        :param query: The user’s question.
        :param top_k: Number of top similar documents to retrieve.
        :return: A RAGResponse object containing the answer, sources, and confidence score.
        """
        # Retrieve relevant chunks
        docs = self.retriever.get_relevant_documents(query)[:top_k]

        # Build context with doc references
        context_str = "\n\n".join(
            f"Document {i+1}:\n{doc.page_content}" for i, doc in enumerate(docs)
        )

        # Format the prompt
        prompt_text = self.enhanced_prompt_template.format(
            context=context_str,
            question=query
        )

        # Get response from the model
        response = self.model([HumanMessage(content=prompt_text)])
        parsed_response = self._parse_rag_response(response.content)

        # Attach source doc metadata
        parsed_response.source_documents = [
            {
                "page_content": doc.page_content,
                "metadata": doc.metadata
            }
            for doc in docs
        ]

        return parsed_response

    def _split_documents(self, docs) -> List[str]:
        """
        Splits loaded documents into smaller chunks using RecursiveCharacterTextSplitter.

        :param docs: List of documents loaded by SimpleDirectoryReader
        :return: List of text chunks
        """
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        split_texts = []
        for doc in docs:
            if hasattr(doc, "text"):
                split_texts.extend(splitter.split_text(doc.text))
        return split_texts

    def _store_in_chroma(
        self,
        text_chunks: List[str],
        embeddings_model
    ) -> Chroma:
        """
        Stores text chunks in Chroma vector database.

        :param text_chunks: List of text segments to store.
        :param embeddings_model: HuggingFaceEmbeddings instance for vector creation.
        :return: A Chroma vector store instance.
        """
        vectorstore = Chroma.from_texts(
            texts=text_chunks,
            embedding=embeddings_model,
            persist_directory="./chroma_storage"
        )
        # Optionally call vectorstore.persist() if you wish to persist on disk
        return vectorstore

    def _parse_rag_response(self, content: str) -> RAGResponse:
        """
        Parses the model's response content to extract a valid JSON for RAGResponse.

        :param content: The model response as a string (possibly containing JSON).
        :return: A RAGResponse object
        """
        try:
            # Attempt to parse the entire response directly as JSON
            data = json.loads(content)
            return RAGResponse(**data)
        except (json.JSONDecodeError, ValidationError):
            # If direct parse fails, try regex-based extraction
            data = self._extract_json(content)
            if not data:
                # Return a fallback RAGResponse if everything fails
                return RAGResponse(
                    answer=(
                        "Unable to parse model response into structured JSON. "
                        f"Original response: {content}"
                    ),
                    confidence_score=0.0
                )
            # Attempt a second parse
            try:
                return RAGResponse(**data)
            except ValidationError:
                return RAGResponse(
                    answer=(
                        "JSON structure doesn't match RAGResponse fields. "
                        f"Original response: {content}"
                    ),
                    confidence_score=0.0
                )

    def _extract_json(self, text: str) -> Dict[str, Any]:
        """
        Attempt to locate a JSON object in the text via regex.

        :param text: The raw text from the model response.
        :return: A dictionary of the found JSON object or an empty dict on failure.
        """
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                pass
        return {}


def main():
    """
    Main entry point to demonstrate usage of the RAGPipeline class.
    Modify the queries or usage as per your requirements.
    """

    # Example usage of the pipeline
    pipeline = RAGPipeline(
        data_dir="/content/data",
        model_name="llama3-8b-8192",
        huggingface_embed_model="sentence-transformers/all-MiniLM-L6-v2",
        groq_api_key=None  # If environment is not set, user will be prompted
    )

    # Example queries
    queries = [
        "What is Chitosan?",
        "Can you summarize the document content?",
    ]

    for q in queries:
        # Basic query with simple textual answer
        basic_answer = pipeline.basic_query(q)
        print(f"Basic Answer for '{q}':\n{basic_answer}\n")

        # Enhanced query with structured output
        structured_resp = pipeline.enhanced_query(q)
        print(f"Enhanced Answer for '{q}': {structured_resp.answer}")
        print(f"Confidence: {structured_resp.confidence_score}")
        print(f"Sources: {len(structured_resp.source_documents)} document(s) retrieved.\n")


if __name__ == "__main__":
    main()
