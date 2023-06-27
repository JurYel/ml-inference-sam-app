FROM public.ecr.aws/lambda/python:3.9

# Copy for requirements.txt 
COPY requirements.txt ./
# RUN python3.9 -m pip install --upgrade pip

RUN python3.9 -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# Copy for source code and other asset

 COPY app.py ./
 COPY model model

# Command can be overwritten by providing a different command in the template directly.
CMD ["app.lambda_handler"]
