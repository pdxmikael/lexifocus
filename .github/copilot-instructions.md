# CONCEPT

The project is called LexiFocus, and is an LLM chat client meant to teach specialized terminology in a second language for students with difficulties learning in traditional classroom settings.

The LLM chat client engages the user in normal conversation around the subject being taught. It fluently switches between the user's native language for in-depth explanations (for example, where the context between the second language term differs slightly with the first language translation, or when feedback needs to be given to the user) and the language being taught. Most interactions will be in the second language.

The application will contain a curriculum, defined as a structured text file, with several different focus areas. Whenever the user demonstrates a new learning by chatting with the LLM about it correctly in the second language, progress will be made and marked as such in the relevant focus area(s).

The LLM will select subjects to converse about from the lesson plan, focusing on areas where the user does not yet exhibit proficiency. It will also attempt to gently guide conversations, through use of follow-up questions, towards areas where the user needs more practice.

**Role of RAG (Retrieval-Augmented Generation):**
The RAG component is crucial for the tutoring aspect. Its primary purposes are:
1.  **Contextual Term Introduction:** Retrieve relevant terms and definitions based on the conversational context. This allows the LLM to naturally introduce and use the target vocabulary the student needs to learn.
2.  **Informed Evaluation:** Provide the LLM with the specific terms and definitions related to the current conversation turn. This context helps the LLM more accurately assess whether the user demonstrates comprehension and correct usage of the terms in the second language.
3.  **Supporting Adaptive Learning:** While topic selection (e.g., via Thompson Sampling) guides the overall focus, the retrieved terms ensure the conversation within that topic remains grounded in the specific vocabulary being taught.

# SOFTWARE STACK

The application is built in LangChain with Chainlit as a frontend.
It uses a structured but human readable text file (e.g. JSON, YAML) for curriculum planning and SQLite for progress tracking.

# DEVELOPMENT METHODOLOGY

The code assistant will take care to abstract and encapsulate functions to ensure components of the system can be developed and tested independently of each other. User testing and feedback will be primary but unit testing can be suitable for specific crucial components, especially if they have no visible frontend.

# ENVIRONMENT

The application is developed on VSCode on Windows within a virtual Python environment (venv).

# DEVELOPMENT PLAN

Current step to implement: #18

1. ~~**Set up Project Structure:** Create the basic project structure for a LangChain/Chainlit application, including `app.py`, configuration files, and directories for data and database.~~
2. ~~**Implement Basic Chainlit App:** In `app.py`, set up a minimal Chainlit application with basic chat functionality using LangChain.~~
3. ~~**Initialize Database:** Add code to initialize the SQLite database (`lexifocus.db`) and create necessary tables (`domain_embeddings`, `activity_log`) when the application starts.~~
4. ~~**Compile Terms:** Create a CSV or YAML file (`terms.yaml` or `terms.csv`) containing 50-100 Swedish economics term definitions.~~
5. ~~**Define Term-to-Topic Mapping:** Ensure the `terms.yaml` file defines how individual terms map to broader learning "topics" used for progress tracking and adaptive selection.~~
6. ~~**Embed and Load Terms:** Implement logic to load term definitions, generate embeddings, and store them in the `domain_embeddings` table.~~
7. ~~**Implement Retrieval:** Query the database for the top 3 relevant term definitions based on context and integrate into the LLM prompt.~~
8. ~~**Implement Core Conversational Chain:** Define and implement the main LangChain conversational chain using user input, chat history, retrieved context, focus topic, and evaluation feedback.~~
9. ~~**Define Logging Function:** Create `activity_log(topic, success)` to record turn outcomes in the SQLite `activity_log` table.~~
10. ~~**Evaluate Turn Success:** Call an evaluation LLM to classify the user's message as progress, setback, or no_change for the selected topic.~~
11. ~~**Log Outcome:** Invoke the logging function with the evaluation result.~~
12. ~~**Incorporate Evaluation Feedback:** Feed the previous turn's evaluation feedback into the next prompt to the main LLM.~~
13. ~~**Create Progress View:** Add a Chainlit UI button to display user progress.~~
14. ~~**Implement Progress Logic:** Query `activity_log` to compute metrics and render them in Chainlit.~~
15. ~~**Implement Adaptive Topic Selection (Initial):** Select the next focus topic using a mastery-threshold (<80%) or round-robin strategy, storing it in session.~~
16. ~~**Modify Prompt for Topic:** Prepend or append a clear "Focus topic: <selected_topic>" line in the LLM prompt.~~
17. **Implement Thompson Sampling:** Replace the mastery-threshold selector with a Thompson Sampling bandit over topics.  
18. **Update Bandit Model:** After each turn's evaluation, update the alpha/beta parameters for the used topic.  
19. **Add Response Mode Control:** Implement in-chat controls (buttons/commands) to adjust LLM verbosity or style.  
20. **Document Application:** Write or update `README.md` with setup and usage instructions.  
21. **(Optional) Visualize Progress:** Enhance progress UI with charts or plots (e.g., via Matplotlib/Plotly).  

# USER TESTING AND HUMAN ITERATION

1. **Test Adaptivity (Initial):** Perform end-to-end testing to verify that the conversation prioritizes topics needing practice.  
2. **Compare Adaptivity:** Conduct side-by-side tests comparing Thompson Sampling vs. mastery-threshold adaptivity.  
3. **Refine Prompt:** Iterate on the system prompt for tone, clarity, and language-switching guidance.  