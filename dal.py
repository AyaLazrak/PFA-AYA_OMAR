import mysql.connector
from mysql.connector import Error
from models import User


class DataBase:
    SERVER_NAME = 'localhost'
    DATABASE_NAME = 'fraude'
    USER = 'root'
    PASSWORD = 'Ayanoorayoub123@'
    Connection = None

    @staticmethod
    def getConnection():
        try:
            if DataBase.Connection is None or not DataBase.Connection.is_connected():
                conn = mysql.connector.connect(
                    host=DataBase.SERVER_NAME,
                    database=DataBase.DATABASE_NAME,
                    user=DataBase.USER,
                    password=DataBase.PASSWORD
                )
                DataBase.Connection = conn
        except Error as e:
            print(f"Error: {e}")
        return DataBase.Connection
    
class UserDao:

    @staticmethod
    def authenticate_user(email: str, password: str) -> User:
        try:
            conn = DataBase.getConnection()
            cursor = conn.cursor()

            cursor.execute("SELECT * FROM login WHERE Email=%s AND Passwordd=%s", (email, password))
            user_data = cursor.fetchone()

            conn.close()

            if user_data:
                return User(*user_data)  # type: ignore
            else:
                return None
        except Error as e:
            print(f"Error: {e}")
            return None

    
    @staticmethod
    def get_all_users() -> list[User]:
        try:
            conn = DataBase.getConnection()
            cursor = conn.cursor()

            cursor.execute("SELECT * FROM login")
            user_data = cursor.fetchall()

            conn.close()

            users = [User(*data) for data in user_data] if user_data else []  # type: ignore
            return users
        except Exception as e:
            print(f"Error: {e}")
            return []

    @staticmethod
    def register_user(user: User) -> str:
        try:
            conn = DataBase.getConnection()
            cursor = conn.cursor()

            cursor.execute("INSERT INTO register (Email, Passwordd) VALUES (%s, %s)",
                           (user.email, user.password))

            conn.commit()
            conn.close()

            return "Registration successful. User data saved to the database."
        except Error as e:
            print(f"Error: {e}")
            return f"Registration failed. Error: {str(e)}"

    @staticmethod
    def get_user(email: str) -> User:
        try:
            conn = DataBase.getConnection()
            cursor = conn.cursor()

            cursor.execute("SELECT * FROM register WHERE Email=%s", (email,))
            user_data = cursor.fetchone()

            conn.close()

            if user_data:
                return User(*user_data)  # type: ignore
            else:
                return None
        except Error as e:
            print(f"Error: {e}")
            return None
