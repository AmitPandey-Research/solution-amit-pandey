import constants
import os
import sys
import glob
from utilities.dataloader import DataLoader

from langchain_community.chat_models import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA, LLMMathChain


os.environ["OPENAI_API_KEY"] = constants.ACCESS


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Error: Please provide a command.")
        sys.exit(1)

    command = sys.argv[1]

    if command == "create_db":
        files = glob.glob("../data/raw/*")
        for file in files:
            if os.path.isfile(file):
                print(f"[INFO] Indexing {file}")
                DataLoader.create_index(file)

    if command == "chat":
        print("-----CHAT STARTED-----")
        # agent_orchestrator_chat()  # Enable when ready
