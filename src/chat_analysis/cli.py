import json
import argparse
import yaml
import os
from tqdm import tqdm
from contextlib import redirect_stdout
from .intent_classification import IntentClassification
from .sentiment_classification import SentimentClassification

def read_config():
    """Read the configuration from config.yaml."""
    with open('config/config.yaml', 'r') as config_file:
        return yaml.safe_load(config_file)

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
    
def print_file_contents(filename):
    """
    Prints the contents of a file.

    Parameters:
        filename (str): The name of the file to be printed.

    Returns:
        None
    """
    with open(filename, 'r') as file:
        contents = file.read()
        print(contents)

    
def main():
    """
    The main function that runs the Chat Analysis CLI tool.

    Parameters:
        None

    Returns:
        None
    """

    # Load the defaults from config.yaml
    defaults = read_config()

    # Create the parser
    parser = argparse.ArgumentParser(description="Chat Analysis CLI tool")

    # Add the arguments
    parser.add_argument('--input_file', type=str, default=defaults.get('input_file', 'data.txt'),
                        help=f'Path to the input file (default: {defaults.get("input_file", "data.txt")})')
    
    parser.add_argument('--output_file', type=str, default=defaults.get('output_file', 'output.txt'),
                        help=f'Path to the input file (default: {defaults.get("output_file", "output.txt")})')
    
    parser.add_argument('--model', type=str, default=defaults.get('model', 'facebook/bart-large-mnli'),
                        help=f'The model for zero-shot classification (default: {defaults.get("model", "facebook/bart-large-mnli")})')

    # Parse the arguments
    args = parser.parse_args()

    chat = read_chat_json(args.input_file)

    # Initialize the classifiers
    intent_classification = IntentClassification(model=args.model)
    sentiment_classification = SentimentClassification(model=args.model)

    print('Chat analysis is starting...')
    
    # Open the output file in write mode
    with open(args.output_file, 'w') as file, redirect_stdout(file):
        # Process the chat
        for conversation in tqdm(chat["conversation"]):
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
                print('-Sentiment: {} ({:.2f}%)'.format(sentiment, s_score*100))
                print('-Intent: {} ({:.2f}%)'.format(intent, i_score*100))

            print('\n')

    print('You can check out the output file for results.\n')
    print_file_contents(args.output_file)

if __name__ == "__main__":
    main()