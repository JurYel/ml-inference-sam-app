import json
import torch
import torch.nn as nn

class NeuralNetwork(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.hidden_layer_1 = nn.Linear(self.in_dim, 20) # input to first hidden layer
        self.hidden_layer_2 = nn.Linear(20, 10)

        self.multiple_layers = nn.Sequential(
            nn.Linear(10, 10),
            nn.Sigmoid(),
            nn.Linear(10, 10),
            nn.Sigmoid(),
            nn.Linear(10, 10),
            nn.Sigmoid(),
            nn.Linear(10, 10),
            nn.Sigmoid(),
            nn.Linear(10, 10),
            nn.Sigmoid()
        )

        self.output_layer = nn.Linear(10, self.out_dim)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x = self.hidden_layer_1(x)
        x = self.activation(x)

        x = self.hidden_layer_2(x)
        x = self.activation(x)

        x = self.multiple_layers(x)

        y = self.output_layer(x)
        y = self.activation(y)

        return y

def lambda_handler(event, context):
    """Sample pure Lambda function

    Parameters
    ----------
    event: dict, required
        API Gateway Lambda Proxy Input Format

        Event doc: https://docs.aws.amazon.com/apigateway/latest/developerguide/set-up-lambda-proxy-integrations.html#api-gateway-simple-proxy-for-lambda-input-format

    context: object, required
        Lambda Context runtime methods and attributes

        Context doc: https://docs.aws.amazon.com/lambda/latest/dg/python-context-object.html
    Returns
    ------
    API Gateway Lambda Proxy Output Format: dict

        Return doc: https://docs.aws.amazon.com/apigateway/latest/developerguide/set-up-lambda-proxy-integrations.html
    """
    #print(event.get("body"))
    # { data: [ x1...x30 ] }
    payload = json.loads(event["body"])["body"]

    payload_data = json.loads(payload)
    print("payload:")
    print(payload_data["data"])
    print("type:")
    print(type(payload_data["data"]))

    # input data
    x = torch.Tensor([payload_data["data"]])
    
    # instantiate the NeuralNetwork model
    model = NeuralNetwork(30, 2)
    
    # load the model
    state = torch.load("model/model.pth")
    model.load_state_dict(state["state_dict"])

    # model prediction
    predictions = model.forward(x).detach().cpu().numpy()[0].tolist()
    print(predictions)
    print(type(predictions))

    res = {
        "predictions": predictions
    }
    
    return {
        "statusCode": 200,
        "body": json.dumps(res),
    }
