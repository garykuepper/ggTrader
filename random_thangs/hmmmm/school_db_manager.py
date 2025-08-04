from pymongo import MongoClient, ASCENDING
from datetime import datetime
import pandas as pd
from bson import ObjectId


class SchoolDBManager:
    def __init__(self, db_name='school', uri='mongodb://localhost:27017/'):
        self.client = MongoClient(uri)
        self.db = self.client[db_name]

        # Collections
        self.students = self.db.students
        self.test_scores = self.db.test_scores

        # Indexes for students
        self.students.create_index('name', unique=True)
        self.students.create_index('student_id', unique=True)

        # Indexes for test_scores
        self.test_scores.create_index(
            [("student_id", ASCENDING), ("test_name", ASCENDING)], unique=True
        )
        self.test_scores.create_index([("date", ASCENDING)])

    # --- Generic converter methods ---

    def docs_to_df(self, docs):
        """
        Convert a list of MongoDB documents (dicts) to a pandas DataFrame.
        Handles ObjectId and datetime conversion.
        """
        if not docs:
            return pd.DataFrame()

        def clean_doc(doc):
            new_doc = {}
            for k, v in doc.items():
                if isinstance(v, ObjectId):
                    new_doc[k] = str(v)
                elif isinstance(v, datetime):
                    new_doc[k] = v
                else:
                    new_doc[k] = v
            return new_doc

        cleaned_docs = [clean_doc(doc) for doc in docs]
        df = pd.DataFrame(cleaned_docs)
        return df

    def df_to_docs(self, df):
        """
        Convert a pandas DataFrame to a list of MongoDB documents (dicts).
        Handles datetime conversion and resets datetime index if present.
        """
        if df.empty:
            return []

        # Reset index if it is a DatetimeIndex or any non-default index
        if isinstance(df.index, pd.DatetimeIndex) or not df.index.equals(pd.RangeIndex(start=0, stop=len(df))):
            df = df.reset_index()

        # Convert datetime columns to Python datetime objects
        for col in df.select_dtypes(include=['datetime64[ns]', 'datetime64[ns, UTC]']).columns:
            df[col] = df[col].apply(lambda x: x.to_pydatetime() if pd.notnull(x) else None)

        docs = df.to_dict(orient='records')
        return docs

    # --- Student methods ---

    def insert_student(self, student):
        try:
            result = self.students.insert_one(student)
            return result.inserted_id
        except Exception as e:
            print(f"Insert student failed: {e}")
            return None

    def update_student(self, student):
        unique_keys = ['name', 'student_id']
        filter_keys = {k: student[k] for k in unique_keys if k in student}
        if not filter_keys:
            raise ValueError("Student must have at least one unique key (name or student_id)")

        update_fields = {k: v for k, v in student.items() if k not in filter_keys}
        if not update_fields:
            print("No fields to update.")
            return None

        result = self.students.update_one(
            filter_keys,
            {'$set': update_fields},
            upsert=True
        )
        return {
            'matched_count': result.matched_count,
            'modified_count': result.modified_count,
            'upserted_id': result.upserted_id
        }

    def find_student(self, filter_dict):
        return self.students.find_one(filter_dict)

    def find_students(self, filter_dict=None):
        if filter_dict is None:
            filter_dict = {}
        docs = list(self.students.find(filter_dict))
        return docs

    def find_students_df(self, filter_dict=None):
        docs = self.find_students(filter_dict)
        return self.docs_to_df(docs)

    def delete_student(self, filter_dict):
        result = self.students.delete_many(filter_dict)
        return result.deleted_count

    # --- Test scores methods ---

    def insert_test_score(self, score):
        try:
            if 'date' in score and isinstance(score['date'], str):
                score['date'] = datetime.fromisoformat(score['date'])
            result = self.test_scores.insert_one(score)
            return result.inserted_id
        except Exception as e:
            print(f"Insert test score failed: {e}")
            return None

    def update_test_score(self, score):
        filter_keys = {
            "student_id": score.get("student_id"),
            "test_name": score.get("test_name")
        }
        if None in filter_keys.values():
            raise ValueError("Both 'student_id' and 'test_name' are required to update test score.")

        update_fields = {k: v for k, v in score.items() if k not in filter_keys}

        if 'date' in update_fields and isinstance(update_fields['date'], str):
            update_fields['date'] = datetime.fromisoformat(update_fields['date'])

        if not update_fields:
            print("No fields to update in test score.")
            return None

        result = self.test_scores.update_one(
            filter_keys,
            {"$set": update_fields},
            upsert=True
        )
        return {
            'matched_count': result.matched_count,
            'modified_count': result.modified_count,
            'upserted_id': result.upserted_id
        }

    def find_test_scores(self, filter_dict=None):
        if filter_dict is None:
            filter_dict = {}
        docs = list(self.test_scores.find(filter_dict))
        return docs

    def find_test_scores_df(self, filter_dict=None):
        docs = self.find_test_scores(filter_dict)
        df = self.docs_to_df(docs)
        if not df.empty and 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            df.sort_index(inplace=True)
        return df

    def delete_test_scores(self, filter_dict):
        result = self.test_scores.delete_many(filter_dict)
        return result.deleted_count

    def get_test_scores_df(self, filter_dict=None):
        # Alias for find_test_scores_df for convenience
        return self.find_test_scores_df(filter_dict)

    def insert_test_scores_df(self, df):
        if df.empty:
            print("DataFrame is empty. Nothing to insert.")
            return 0

        records = self.df_to_docs(df)

        try:
            result = self.test_scores.insert_many(records)
            inserted_count = len(result.inserted_ids)
            print(f"Inserted {inserted_count} test score records.")
            return inserted_count
        except Exception as e:
            print(f"Failed to insert test scores: {e}")
            return 0
