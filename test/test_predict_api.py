import requests

# Define the API endpoint
url = "http://127.0.0.1:8000/predict/"

# Test cases
test_data = [
    {
        "context": "Arthur's Magazine (1844–1846) was an American literary periodical published in Philadelphia in the 19th century.First for Women is a woman's magazine published by Bauer Media Group in the USA.",
        "prompt": "Which magazine was started first Arthur's Magazine or First for Women?",
        "response": "First for Women was started first.",
        "expected_prediction": "yes"
    },
    {
        "context": "The Oberoi family is an Indian family that is famous for its involvement in hotels, namely through The Oberoi Group.The Oberoi Group is a hotel company with its head office in Delhi.",
        "prompt": "The Oberoi family is part of a hotel company that has a head office in what city?",
        "response": "Delhi",
        "expected_prediction": "no"
    },
    {
        "context": "Allison Beth \"Allie\" Goertz (born March 2, 1991) is an American musician. Goertz is known for her satirical songs based on various pop culture topics. Her videos are posted on YouTube under the name of Cossbysweater.Milhouse Mussolini van Houten is a fictional character featured in the animated television series \"The Simpsons\", voiced by Pamela Hayden, and created by Matt Groening who named the character after President Richard Nixon's middle name.",
        "prompt": "Musician and satirist Allie Goertz wrote a song about the \"The Simpsons\" character Milhouse, who Matt Groening named after who?",
        "response": "President Richard Nixon",
        "expected_prediction": "no"
    },
    {
        "context": "The Oberoi family is an Indian family that is famous for its involvement in hotels, namely through The Oberoi Group.The Oberoi Group is a hotel company with its head office in Delhi.",
        "prompt": "The Oberoi family is part of a hotel company that has a head office in what city?",
        "response": "lahore",
        "expected_prediction": "no"
    },
    {
        "context": "The Oberoi family is an Indian family that is famous for its involvement in hotels, namely through The Oberoi Group.The Oberoi Group is a hotel company with its head office in Delhi.",
        "prompt": "The Oberoi family is part of a hotel company that has a head office in what city?",
        "response": "Delhi",
        "expected_prediction": "yes"
    }
    # Add more examples here as needed
]

# Iterate through test cases and send POST requests
for idx, data in enumerate(test_data, 1):
    payload = {
        "context": data["context"],
        "prompt": data["prompt"],
        "response": data["response"]
    }
    
    response = requests.post(url, json=payload)
    
    if response.status_code == 200:
        result = response.json()
        print(f"Test Case {idx}:")
        print(f"Payload: {payload}")
        print(f"Expected: {data['expected_prediction']}, Actual: {result.get('prediction')}\n")
    else:
        print(f"Test Case {idx} failed with status code {response.status_code}")
        print(f"Response: {response.text}\n")
