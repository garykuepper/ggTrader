from datetime import datetime
import pandas as pd
from school_db_manager import SchoolDBManager
# Assuming SchoolDBManager class is imported or defined in the same file

def main():
    manager = SchoolDBManager()

    # Insert a student
    student = {
        "name": "Olivia",
        "student_id": "S12345",
        "age": 11,
        "grade": "6th",
        "favoriteColor": "red"
    }
    inserted_id = manager.insert_student(student)
    print(f"Inserted student ID: {inserted_id}")

    # Update student
    student_update = {
        "name": "Olivia",
        "student_id": "S12345",
        "grade": "7th",
        "favoriteColor": "blue"
    }
    update_result = manager.update_student(student_update)
    print(f"Update student result: {update_result}")

    # Insert a test score
    test_score = {
        "student_id": "S12345",
        "test_name": "Math Midterm",
        "score": 88,
        "date": datetime(2024, 4, 15)
    }
    inserted_score_id = manager.insert_test_score(test_score)
    print(f"Inserted test score ID: {inserted_score_id}")

    # Update test score
    test_score_update = {
        "student_id": "S12345",
        "test_name": "Math Midterm",
        "score": 92  # updated score
    }
    update_score_result = manager.update_test_score(test_score_update)
    print(f"Update test score result: {update_score_result}")

    # Find student
    found_student = manager.find_student({"name": "Olivia"})
    print(f"Found student: {found_student}")

    # Find test scores for a student
    scores = manager.find_test_scores({"student_id": "S12345"})
    print(f"Test scores for student S12345: {scores}")

    # Demonstrate retrieving test scores as a pandas DataFrame
    df_scores = manager.get_test_scores_df({"student_id": "S12345"})
    print("\nTest scores DataFrame:")
    print(df_scores)

    # Demonstrate inserting test scores from a DataFrame
    new_scores_data = {
        "student_id": ["S12345", "S12345"],
        "test_name": ["Science Quiz", "History Quiz"],
        "score": [85, 90],
        "date": [datetime(2024, 4, 20), datetime(2024, 4, 22)]
    }
    df_new_scores = pd.DataFrame(new_scores_data)
    inserted_count = manager.insert_test_scores_df(df_new_scores)
    print(f"Inserted {inserted_count} new test scores from DataFrame.")

    # Find updated test scores for the student
    updated_scores = manager.find_test_scores({"student_id": "S12345"})
    print(f"\nUpdated test scores for student S12345: {updated_scores}")

    # Delete test scores for a student
    deleted_count = manager.delete_test_scores({"student_id": "S12345"})
    print(f"Deleted {deleted_count} test score(s)")

    # Delete student
    deleted_students = manager.delete_student({"student_id": "S12345"})
    print(f"Deleted {deleted_students} student(s)")

if __name__ == "__main__":
    main()
