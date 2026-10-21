# Multi-Domain Task-Oriented Chatbot with Retrieval (MultiWOZ)

A multi-domain dialogue system trained on the [MultiWOZ 2.2](https://github.com/budzianowski/multiwoz) dataset. Given a user message, the chatbot detects the domain (hotel, restaurant, train, …), extracts the user's intent and slot values, retrieves matching database entries with vector search, and generates a reply grounded in the retrieved entries and the conversation history.

The system can be used through a **Streamlit chat interface** or a **command-line chatbot**.

1. **Domain classification:** a DistilBERT model, trained on user messages from MultiWOZ, decides which domain each message belongs to (hotel, restaurant, train, …).
2. **Intent and slot extraction:** intents and slot values from the MultiWOZ schema are encoded with `all-MiniLM-L6-v2` sentence-transformer embeddings and stored in per-domain FAISS indices. The user message is matched to the closest intent and slots by cosine similarity, with confidence thresholds.
3. **Database retrieval:** MultiWOZ database entries are converted to text, embedded and indexed with FAISS (normalised inner product, i.e. cosine similarity). The most relevant entries for the query are retrieved for the generator. A LangChain retriever wrapper (`retrieval_adv.py`) exposes the same retrieval through the LangChain interface.
4. **Response generation:** a fine-tuned T5 model generates the reply from a prompt combining the conversation history, user message, domain, intents, slots and retrieved database entries. `generator.py` also fine-tunes DistilGPT-2 as an alternative generator; the prompt formats we tried are in `prompts.txt`.
5. **Conversation state:** `state_manager.py` tracks the domain, intents, slots, retrieved results and dialogue history across turns.

## Team

Developed by Fedor Chikhachev and Vladyslav Siulhin for the NLP course at the University of Luxembourg.

- **Fedor Chikhachev:** data preprocessing, schema parsing, DistilBERT domain classifier, response generation, conversation state tracking
- **Vladyslav Siulhin:** sentence-transformer embeddings and FAISS indices, intent and slot extraction, database retrieval, LangChain retriever wrapper, Streamlit chat interface

