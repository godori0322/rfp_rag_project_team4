from config import Config
from chatbot import Chatbot

def main():
    user_id = input("사용자 아이디를 입력하세요: ")
    bot = Chatbot(user_id=user_id)
    
    print(f"\n'{user_id}'님, 안녕하세요! RAG 챗봇입니다. (종료하시려면 'exit' 또는 'quit'을 입력하세요)\n")

    if bot.history:
        print("[이전 대화 기록]")
        for i, msg in enumerate(bot.history):
            role = "사용자" if msg.type == 'human' else "AI"
            print(f"{i+1}. {role}: {msg.content}")
        print("-" * 20)

    while True:
        query_str = input("질문을 입력하세요: ")
        if query_str.lower() in ["exit", "quit"]:
            print("챗봇을 종료합니다.")
            break
        
        response = bot.ask(query_str)
        print("\n[답변]")
        print(response['answer'])
        print("-" * 20)

if __name__ == "__main__":
    main()