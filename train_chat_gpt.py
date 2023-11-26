from openai import OpenAI
import time
client = OpenAI()

# client.files.create(
#   file=open("commentary_train_prepared.json", "rb"),
#   purpose="fine-tune"
# )

# client.fine_tuning.jobs.create(
#   training_file="file-PoXR70mrvksITiCwM0RsQLGH", 
#   model="gpt-3.5-turbo"
# )

# time.sleep(5)
# print(client.fine_tuning.jobs.list(limit=10))
while True:
    for events in client.fine_tuning.jobs.list_events(fine_tuning_job_id="ftjob-mlyDK5bYHlcM10C9ZT0TlIhn", limit=1):
        print(events)
    time.sleep(10)