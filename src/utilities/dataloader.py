from typing import Dict, Any, List
import re
import pandas as pd
from langchain.document_loaders import TextLoader
from langchain.schema import Document
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter

class DataLoader:
    @staticmethod
    def parse_engine_specs(engine_str: str) -> Dict[str, Any]:
        engine_match = re.match(r"(\w+),\s*(\d+)\s*HP,\s*(\d+)\s*cc", engine_str, re.IGNORECASE)
        if engine_match:
            engine_type, hp, cc = engine_match.groups()
            return {
                "engine_type": engine_type,
                "horse_power": int(hp),
                "cc": int(cc)
            }
        return {}

    @staticmethod
    def parse_other_specs(specs_str: str) -> Dict[str, Any]:
        body_type_match = re.match(r"(\w+)", specs_str)
        mileage_match = re.search(r"(\d+)\s*km/l", specs_str)
        speed_match = re.search(r"(\d+)\s*km/h", specs_str)

        return {
            "body_type": body_type_match.group(1) if body_type_match else None,
            "mileage_kmpl": int(mileage_match.group(1)) if mileage_match else None,
            "top_speed_kmph": int(speed_match.group(1)) if speed_match else None
        }

    @staticmethod
    def process_documents(file_path: str) -> List[Document]:
        file_path = Path(file_path)
        name = file_path.stem

        if file_path.suffix == ".csv":
            df = pd.read_csv(file_path)
            docs = []
            for _, row in df.iterrows():
                engine = DataLoader.parse_engine_specs(row.get("Engine Specifications", ""))
                specs = DataLoader.parse_other_specs(row.get("Other Specifications", ""))

                page_content = f"""
                    description: {row.get('Description', '')},
                    car_name: {row.get('Car Name')},
                    manufacturer: {row.get('Manufacturer')},
                    launch_year: {row.get('Launch Year')},
                    engine_type: {engine.get('engine_type')},
                    horse_power: {engine.get('horse_power')},
                    cc: {engine.get('cc')},
                    body_type: {specs.get('body_type')},
                    mileage_kmpl: {specs.get('mileage_kmpl')},
                    top_speed_kmph: {specs.get('top_speed_kmph')},
                    user_ratings: {row.get('User Ratings')},
                    ncap_rating: {row.get('NCAP Global Rating')}
                """.strip()

                metadata = {
                    "car_name": row.get("Car Name"),
                    "manufacturer": row.get("Manufacturer"),
                    "launch_year": row.get("Launch Year"),
                    "engine_type": engine.get('engine_type'),
                    "horse_power": engine.get('horse_power'),
                    "cc": engine.get('cc'),
                    "body_type": specs.get('body_type'),
                    "mileage_kmpl": specs.get('mileage_kmpl'),
                    "top_speed_kmph": specs.get('top_speed_kmph'),
                    "user_ratings": float(row.get("User Ratings", 0)),
                    "ncap_rating": int(row.get("NCAP Global Rating", 0))
                }

                docs.append(Document(page_content=page_content, metadata=metadata))

            output_dir = Path(__file__).resolve().parent.parent.parent / "data" / "processed"
            output_dir.mkdir(parents=True, exist_ok=True)
            file_output = output_dir / f"{name}.txt"

            with file_output.open("w", encoding="utf-8") as f:
                for doc in docs:
                    f.write(f"{doc.page_content}\n\n")

            return docs

        else:
            loader = TextLoader(file_path)
            documents = loader.load()

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=500,
                separators=["\n### ", "\n## ", "\n# ", "\n\n", "\n", " ", ""],
                length_function=len,
            )
            chunks = text_splitter.split_documents(documents)
            return chunks

    @staticmethod
    def create_index(file_path: str):
        docs = DataLoader.process_documents(file_path)
        index_name = Path(file_path).stem
        embeddings = SentenceTransformerEmbeddings(model_name="all-mpnet-base-v2")
        vectordb = FAISS.from_documents(docs, embeddings)
        vectordb.save_local(index_name)
        print(f"Index files for {index_name} saved.")

if __name__ == "__main__":
    country_file_path = "../../data/raw/country_data.md"
    DataLoader.create_index(str(country_file_path))