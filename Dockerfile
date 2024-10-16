FROM public.ecr.aws/lambda/python:3.11 AS builder
# Set the working directory
WORKDIR /var/task

# Copia tu código
COPY pictoconecta_textgeneration_api_5.py .
COPY requirements.txt .
COPY config.py .
COPY historial.json .
COPY utils.py .

# Install dependencies from the requirements file
RUN pip install --no-cache-dir -r requirements.txt

# Command to run your app using the Lambda handler
CMD ["pictoconecta_textgeneration_api_5.lambda_handler"]

