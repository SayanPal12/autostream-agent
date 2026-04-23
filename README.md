# AutoStream AI Agent

## Project Overview

AutoStream AI Agent is a conversational AI system built for a fictional SaaS company named **AutoStream**, which provides automated video editing tools for content creators. The objective of this project is to convert normal user conversations into qualified business leads through an intelligent multi-step workflow.

The agent can greet users, answer pricing and policy questions using Retrieval-Augmented Generation (RAG), detect high purchase intent, collect lead details, and trigger a mock backend API after qualification.

---

## Key Features

* Intent Detection using structured LLM output
* RAG-based pricing / feature / policy responses
* High-intent lead qualification workflow
* Stateful multi-turn memory
* Mock lead capture API integration
* LangGraph workflow orchestration
* Conversation history management

---

## Tech Stack

* Python
* LangGraph
* LangChain
* Groq LLM (`openai/gpt-oss-120b`)
* Google Embeddings (`gemini-embedding-001`)
* Chroma Vector Database
* Pydantic Structured Output

---

## Project Structure

```text
AutoStream-Agent/
│── Agent.py
│── Run.py
│── SystemPrompts.py
│── pricing.md
│── chroma_db/
│── requirements.txt
│── README.md
```

---

## How to Run the Project Locally

### 1. Clone Repository

```bash
git clone <your-repo-link>
cd AutoStream-Agent
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Create Environment File

Create a `.env` file and add your API keys:

```env
GROQ_API_KEY=your_key_here
GOOGLE_API_KEY=your_key_here
```

### 4. Run the Application

```bash
python Run.py
```

### 5. Start Chatting


## Architecture Explanation

I selected **LangGraph** because this project requires controlled multi-step workflows rather than a simple chatbot. The user journey can branch into two separate paths: normal support conversations or high-intent lead qualification. LangGraph is ideal for this because it allows stateful node-based routing and clear transitions between tasks. 

The graph starts with an **intent detection node**, where the LLM returns structured output as either `low_intent` or `high_intent`. Low-intent users are routed to the normal chat node, which uses tool-calling with a custom RAG tool connected to a Chroma vector store built from `pricing.md`. This enables grounded answers for plans, pricing, and company policies. 

High-intent users are routed to a separate qualification node. This node uses structured output to extract and store three required fields: **name**, **email**, and **creator platform**. Missing fields are collected step-by-step across turns. Once all fields are available, a mock API function is triggered to simulate lead capture. 

State is managed through LangGraph state memory using `InMemorySaver`, which preserves conversation thread data such as message history, intent, and collected lead information. Old messages are trimmed to reduce token usage while keeping recent context active. 

---

## WhatsApp Deployment Using Webhooks

To deploy this agent on WhatsApp, I would use the **WhatsApp Business API** through a provider such as **Twilio** or **Meta Cloud API**.

### Workflow

1. A user sends a message to the AutoStream WhatsApp number.
2. WhatsApp forwards the message to a backend webhook endpoint such as:

```text
POST /webhook
```

3. The backend receives:

* user phone number
* message text
* timestamp

4. The phone number is used as a unique session ID to load user state and memory.

5. The incoming message is passed to the LangGraph agent:

* Intent Detection
* RAG Q&A
* Lead Qualification
* Tool Execution

6. The generated response is sent back through the WhatsApp API to the user.

7. If lead details are completed, they are stored in a database / CRM / Google Sheets for the sales team.

### Production Improvements

* Redis / PostgreSQL for persistent memory
* Webhook signature verification
* Rate limiting
* Human handoff support
* Logging and analytics
* Retry handling for failed deliveries

---

## Demo Video


```text
https://drive.google.com/file/d/1xXc0yCIZoARymWs1bCLlDKZnPkxcgNxF/view?usp=sharing
```


---

## Future Improvements

* Streamlit Web UI
* Real CRM Integration
* WhatsApp Live Deployment
* Better lead scoring
* Analytics dashboard
* Multilingual support

---
