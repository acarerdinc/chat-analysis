import json
import argparse
import yaml
import os
import logging
from contextlib import redirect_stdout
from .intent_classification import IntentClassification
from .sentiment_classification import SentimentClassification

# Configure the logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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

    logger.info('Chat analysis is starting...')
    
    # Process the chat
    for conversation in chat["conversation"]:
        role = conversation["role"]
        message = conversation["message"]

        # Logging the role and message
        logger.info('[{}]'.format(role.upper()))
        logger.info(message)

        if role == "customer":
            # classify the sentiment of the message
            sentiment, s_score = sentiment_classification.classify(message)

            # classify the intent of the message
            intent, i_score = intent_classification.classify(message)

            # Logging the results
            logger.info('-Sentiment: {} ({:.2f}%)'.format(sentiment, s_score*100))
            logger.info('-Intent: {} ({:.2f}%)'.format(intent, i_score*100))

            # Set the json output
            conversation['sentiment'] = sentiment
            conversation['sentiment_score'] = s_score
            conversation['intent'] = intent
            conversation['intent_score'] = i_score

    # Logging the completion message
    logger.info('You can check out the output file for results.\n')
    
    # Write the results to the output file
    with open(args.output_file, 'w') as outfile:
        json.dump(chat, outfile, indent=4)


if __name__ == "__main__":
    main()