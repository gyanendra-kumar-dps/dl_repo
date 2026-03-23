import nltk
from nltk.chat.util import Chat,reflections
from datetime import datetime
reflections={
    "I am":"You are",
    "I was":"You were",
    "I":"You",
    "I'm":"You are",
    "I will":"You wil",
    "I can":"You can",
    "I had":"You had",
    "I could":"You could",
    "I'll":"You will",
    "I know":"You know",
    "Your":"My",
    "Me":"You"
}
pairs=[
    [
        "Hi",
        ["Hey there!","What's good"]
    ],
    [
        "Who is your favourite footballer",
        ["Messi","Ronaldo","Neymar"]
    ],
    [
        "Who is you favourite cricketer",
        ["Dhoni",'Virat',"Rohit"]
    ],
    [
        "Who created you",
        ["People at google created me"]
    ],
    [
        "Hi (.*)",
        ["Hey! there, How are you."]
    ],
    [
        "My name is (.*)",
        ["Hello %1, How are you today."]
    ],
    [
        "What is your name",
        ["My name is jarvis. I am created by google"]
    ],
    [
        "Sorry (.*)",
        ["It's all right"]
    ],
    [
        "(.*) your age.",
        ["I do not have an age."]
    ],
    [
        "(.*) time",
        [f"The time is {datetime.now().strftime("%H:%M:%S")}"]
    ],
    [
        "(.*) (city|location)",
        ["I am based in Bangalore,Karnataka,India,Asia"]
    ]
]
def chat():
    print("Hi i am an AI chatbot and how can i assist you today.")
    cur_chat=Chat(pairs,reflections)
    cur_chat.converse()
if __name__=="__main__":
    chat()