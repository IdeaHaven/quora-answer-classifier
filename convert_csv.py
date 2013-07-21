import csv
import json

f = open('answered_data_10k.txt')
lines = f.readlines()
f.close()

f = csv.writer(open("test.csv", "wb+"))

f.writerow(["question_text", "followers", "name", "topicFollowers", "answer", "anonymous"])

for x in lines:
	data = json.loads(x)
	try:
	   f.writerow([data["question_text"], 
	       data["context_topic"]["followers"], 
	       data["context_topic"]["name"], 
	       data["question_key"],
	       data["anonymous"],
	       data["__ans__"]])
	except:
		pass

# for x in 
# data = json.loads("answered_data_10k.json")
# print data
# 
# with open("answered_data_10k.json") as file:
#	  data = json.loads(file)
#	  print data
# 
# f = csv.writer(open("test.csv", "wb+"))
# 
# # Write CSV Header, If you dont need that, remove this line
# f.writerow(["question_text", "followers", "name", "topicFollowers", "topicName", "questionKey", "answer", "anonymous"])
# 
# for x in data:
# 	print x
# 	f.writerow([x["question_text"], 
#				  x["context_topic"]["followers"], 
#				  x["context_topic"]["name"], 
#				  x["question_key"],
#				  x["anonymous"],
#				  x["__ans__"]])