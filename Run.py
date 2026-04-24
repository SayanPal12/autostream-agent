from Agent import chatbot
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

config={"configurable": {"thread_id": "user1"}}
while True:
    user_input= input("Enter ....")
    if user_input.lower() =="exit" or user_input.lower() =="bye":
        print("User: exit")
        print("AI: Bye.")
        break
    result= chatbot.invoke(
        {
            "user_input":user_input,
            "messages": [],
            'name':None,
            'email': None,
            'platform':None

        }, config=config
    )
    if(result['intent']=='high_intent'):
        continue
    print("User:", user_input)
    ai_last_message= result["messages"][-1]
    print("AI: ", ai_last_message.content)

