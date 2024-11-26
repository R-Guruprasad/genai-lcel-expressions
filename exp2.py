from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

# Define the prompt template with two parameters
prompt = PromptTemplate(
    input_variables=["topic", "context"],
    template="Write a summary about {topic} based on the following context: {context}"
)

# Initialize the language model
llm = OpenAI(model="text-davinci-003", temperature=0.7)

# Define the response schema for the output parser
response_schemas = [
    ResponseSchema(name="summary", description="A concise summary of the topic"),
    ResponseSchema(name="key_points", description="Main points covered in the summary")
]

output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

# Example function to evaluate the LCEL expression
def evaluate_expression(topic, context):
    # Generate the prompt with input parameters
    formatted_prompt = prompt.format(topic=topic, context=context)
    
    # Get the response from the LLM
    raw_output = llm(formatted_prompt)
    
    # Parse the output into a structured format
    parsed_output = output_parser.parse(raw_output)
    
    return parsed_output

# Example usage
if __name__ == "__main__":
    topic = "Artificial Intelligence"
    context = """Artificial Intelligence is a field of study focusing on creating machines
               capable of mimicking human intelligence.It includes machine learning, robotics,
                and natural language processing."""
    result = evaluate_expression(topic, context)
    print("LCEL Expression Output:")
    print(result)