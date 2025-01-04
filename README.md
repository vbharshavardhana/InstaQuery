# 📖 InstaQuery: A Real-Time, Page-Aware Chatbot
**InstaQuery** is a cutting-edge real-time chatbot that uses LangChain, LLaMA models, and Google's Gemini API to provide intelligent, context-aware responses. It features a user-friendly UI built with Streamlit and a robust backend powered by FastAPI. The project includes Retrieval-Augmented Generation (RAG) for enhanced information retrieval and leverages CouchDB for efficient data storage.

---

## 📋 Table of Contents
- [About the Project](#about-the-project)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Installation](#installation)
- [Usage](#usage)
- [Project Workflow](#project-workflow)
- [Future Enhancements](#future-enhancements)
- [Contributing](#contributing)
- [License](#license)

---

## 🔍 About the Project
InstaQuery is designed to offer **multi-turn conversational capabilities** with **page-aware context retrieval**. It aims to enhance user engagement by delivering **accurate responses** based on the page content, making it ideal for real-time customer support and knowledge-based applications.

---

## 🚀 Features
- **Page-aware chatbot** using Retrieval-Augmented Generation (RAG)
- **Real-time conversations** with multi-turn interactions
- **Efficient NLP pipeline** using LangChain
- **Responsive UI** built with Streamlit
- **Fast and scalable API** using FastAPI
- **Database integration** with CouchDB
- **Google Gemini API integration** for high-quality LLM responses

---

## 🛠 Tech Stack
- **Python** (Backend logic and API)
- **Streamlit** (Frontend UI)
- **FastAPI** (Backend API handling)
- **LangChain** (NLP pipeline and conversational AI)
- **Google Gemini API** (LLM responses)

---

## 📥 Installation
```bash
# Clone the repository
git clone [repo.git]

# Navigate to the project directory
cd instaquery

# Install the required dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

---

## 📚 Usage
1. **Run the backend API:**  
   Start the FastAPI server by running `uvicorn main:app --reload`.
   
2. **Launch the Streamlit app:**  
   Run `streamlit run app.py` to start the UI.

3. **Interact with the chatbot:**  
   Use the page-aware chatbot to get contextually relevant responses in real-time.

---

## 🔄 Project Workflow
Here’s a detailed breakdown of the project workflow:

1. **User Input:**
   - The user enters a query via the Streamlit UI.

2. **Frontend Processing:**
   - The UI sends the query to the FastAPI backend using an API request.

3. **Backend Logic (FastAPI):**
   - FastAPI receives the query and forwards it to the LangChain pipeline for processing.
   - The chatbot fetches relevant context using RAG from CouchDB (if available).

4. **NLP Pipeline (LangChain):**
   - The query is processed through a pre-trained llm model.
   - If additional context is needed, LangChain leverages RAG to retrieve related data from the BM25.

5. **External API Call:**
   - The chatbot queries the Google Gemini API for an LLM-powered response.

6. **Response Generation:**
   - The backend assembles the final response by combining the LLM output and retrieved context.

7. **Frontend Display:**
   - The processed response is sent back to the Streamlit UI for the user to see.

---

## 📈 Future Enhancements
- 🔄 **Improve NLP pipeline** by using a more efficient sentence transformer model.
- 🧩 **Integrate LangChain for better multi-turn conversations.**
- 🎨 **Refine UI to look similar to ChatGPT for a seamless user experience.**
- 📊 **Add analytics dashboard** to monitor user interactions and performance.

---

## 🤝 Contributing
Contributions are welcome! Please fork the repository and submit a pull request for review.

---

## 📜 License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---
