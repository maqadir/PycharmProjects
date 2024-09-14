import requests
import json
import jsonpath
from jsonpath_ng.ext import parse



url = " https://fake-json-api.mock.beeceptor.com/users"
response = requests.get(url)
#print(response)
#print(response.content)
#print(response.headers)

#Parse response to JSON format
json_response = json.loads(response.text)
print(json_response)

#Fetch value using JSON Path
#jsonpath_expr = parse('[*].name')
#print(jsonpath_expr)
# Execute the query
#result = [match.value for match in jsonpath_expr.find(json_response)]

# Print the result
#print(result)

names = jsonpath.jsonpath(json_response,'$[*].name')

print(names[0])

assert names[0] == 'Agnes Schoen'