# Author: Amit Pandey

# solution-amit-pandey
RAG Bot for Specialized Question-Answering

Current [as of 15th of May 2025] main working branch is: 'main_working_branch'.

Instructions: 

1. Create a new virtual environment and install all the dependencies using `pip install -r requirements.txt`
2. Activate the virtual environment.
3. Change directory to `solution-amit-pandey\src`
4. `[Not Required]` Create vector database from the raw files using `python main.py create_db` - I have already provided the vector databases after processing the raw files.
5. Add OpenAI API Key as `ACCESS` variable in `solution-amit-pandey\src\constants.py` 
6. Run the agent using `python main.py`
7. Type your query and hit enter to get response from one of the following agents: Car Agent, Country Agent, Math Agent, Car_Math Agent, Custom Code Agent.
8. Type exit to exit the chat.
9. Custom Code Agent saves the python scripts under solution-amit-pandey\src\utilities\math_scripts. You can view and edit the scripts.
