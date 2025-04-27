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

Current step to implement: #16

1.  ~~**Set up Project Structure:** Create the basic project structure for a LangChain/Chainlit application, including `app.py` (or similar main file), configuration files, and directories for data and database.~~
2.  ~~**Implement Basic Chainlit App:** In `app.py`, set up a minimal Chainlit application with basic chat functionality using LangChain.~~
3.  ~~**Initialize Database:** Add code to initialize the SQLite database (`lexifocus.db`) and create necessary tables (`domain_embeddings`, `activity_log`) when the application starts.~~
4.  ~~**Compile Terms:** Create a CSV or YAML file (`terms.yaml` or `terms.csv`) containing 50-100 Swedish economics term definitions.~~
5.  ~~**Define Term-to-Topic Mapping:** Ensure the `terms.yaml` file defines how individual terms map to broader learning "topics" used for progress tracking and adaptive selection.~~
6.  ~~**Embed and Load Terms:** Implement logic (e.g., in a setup function or on startup) to read the term definitions, generate embeddings using a suitable model (e.g., from LangChain Embeddings), and store both the terms and their embeddings in the `domain_embeddings` table.~~
7.  ~~**Implement Retrieval:** In the main chat logic, before calling the LLM, add code to query the SQLite database for the top 3 most relevant term definitions based on the current chat history/context, using vector similarity search on the embeddings. Integrate these retrieved terms into the context provided to the LLM *to facilitate contextual term introduction and support evaluation*.~~
8.  ~~**Implement Core Conversational Chain:** Define and implement the main LangChain conversational chain. This chain will take user input, chat history, retrieved context (from Step 7), the current focus topic (from Step 16/19), and evaluation feedback (from Step 12) to generate the tutor's response.~~
9.  ~~**Define Logging Function:** Create a helper function `activity_log(topic, success)` that takes a topic name and a boolean success indicator, and inserts a corresponding record into the `activity_log` table in SQLite.~~
10. ~~**Evaluate Turn Success:** After receiving the user's message and before generating the main response, make a separate call to the configured LLM. This call should analyze the user's last message in the context of the `selected_topic_for_turn`. The LLM's task is to determine if the user demonstrated **progress (success)**, **setback (failure)**, or **no significant change** regarding the topic.~~
11. ~~**Log Outcome:** Call the `activity_log` function with the determined topic and the success/failure outcome from the previous step.~~
12. ~~**Incorporate Evaluation Feedback:** Incorporate brief feedback based on the evaluation result (progress/setback/no_change from Step 10) into the system prompt or context provided to the main LLM (Step 8) for the *next* chat turn. This helps guide the LLM's response style or focus.~~
13. ~~**Create Progress View:** Add a button within the Chainlit UI to display user progress.~~
14. ~~**Implement Progress Logic:** Code the logic behind the progress view to query the `activity_log` table, calculate progress metrics (like percentage accuracy per topic), and display this data in the Chainlit interface.~~
15. ~~**Implement Adaptive Topic Selection (Initial):** Before the main LLM call (Step 8), add logic to select the next topic. *Handle initial selection (e.g., random or first topic) if no progress data exists.* Otherwise, query progress data (via `activity_log`) and use a simple strategy: prioritize topics with mastery below 80%. If all topics are above 80%, use a round-robin approach. Store the selected topic (e.g., in user session state).~~
16. **Modify Prompt for Topic:** Append a string like "Focus topic: <selected_topic>" to the system prompt or context sent to the main LLM (Step 8), using the topic selected in the previous step.
17. **Test Adaptivity (Initial):** Perform end-to-end testing to ensure the chat conversation consistently focuses on topics where the user needs more practice, based on the tracked mastery data.
18. **Implement Thompson Sampling:** Replace the mastery threshold topic selection logic (Step 15) with a Thompson Sampling algorithm. Use the success/failure counts per topic from the `activity_log` table as input for the bandit model (alpha/beta parameters for Beta distribution).
19. **Update Bandit Model:** After evaluating turn success (Step 10) and logging the outcome (Step 11), update the internal state (alpha/beta counts) of the Thompson Sampling model for the relevant topic.
20. **Compare Adaptivity:** Conduct tests to compare the effectiveness and speed of topic adaptation using the Thompson Sampling approach versus the simple mastery threshold approach.
21. **Refine Prompt:** Review and adjust the system prompt or instructions provided to the main LLM (Step 8) to ensure clarity, an empathetic tone, appropriate academic style for language learning, *and clear guidance on when to switch between the native (English) and target (Swedish) languages*.
22. **Add Response Mode Control:** Implement logic within Chainlit (e.g., using `cl.Action` buttons or chat commands like `/set_mode concise`) that allow the user to control the verbosity or style of the LLM's responses. Modify the LLM call parameters or prompt accordingly based on the selected mode.
23. **Document Application:** Create or update a `README.md` file explaining how to set up, configure, and run the LexiFocus application.
24. **(Optional) Visualize Progress:** Enhance the progress view using Chainlit's visualization elements or by integrating a library like Matplotlib/Plotly if needed.