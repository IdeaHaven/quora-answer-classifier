# Enter your code here. Read input from STDIN. Print output to STDOUT
# Enter your code here. Read input from STDIN. Print output to STDOUT
import sys
import json

count = 0
for line in sys.stdin:
    try:
        data = json.loads(line)
        if int(data):
            count = count + 1
    except:
        pass
    if count == 2:
        data = json.loads(line)
        try:
            if int(data):
                pass
            else:
                print data
        except:
            try:
                jsonString = json.JSONEncoder().encode({
                    "__ans__": bool(1), 
                    "question_key": data["question_key"]
                })
                sys.stdout.write(jsonString + "\n")
            except:
                pass