import json
import argparse
from intent_classification import IntentClassification
from sentiment_classification import SentimentClassification

def read_chat_json(file_path):
    """
    Read a JSON file and return its contents.

    Args:
        file_path (str): The path to the JSON file.

    Returns:
        data: The contents of the JSON file.
    """
    with open(file_path) as json_file:
        data = json.load(json_file)
        return data

if __name__ == "__main__":

    # Create the parser
    parser = argparse.ArgumentParser(description="Chat Analysis CLI tool")

    # Add the arguments
    parser.add_argument('--input_file', type=str, default="data/sample-data-2.json",
                        help='Path to the input file (default: data/sample-data-2.txt)')

    # Parse the arguments
    args = parser.parse_args()

    chat = read_chat_json(args.input_file)

    # Initialize the classifiers
    intent_classification = IntentClassification()
    sentiment_classification = SentimentClassification()
    
    # Process the chat
    for conversation in chat["conversation"]:
        role = conversation["role"]
        message = conversation["message"]
        
        # print the role and message
        print('[{}]'.format(role.upper()))
        print(message)

        if role == "customer":
            # classify the sentiment of the message
            sentiment, s_score = sentiment_classification.classify(message)

            # classify the intent of the message
            intent, i_score = intent_classification.classify(message)

            # print the results
            print('Sentiment: {} ({:.2f}%)'.format(sentiment, s_score*100))
            print('Intent: {} ({:.2f}%)'.format(intent, i_score*100))

        print('\n')
