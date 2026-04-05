from notifications import send_slack_alert

result = send_slack_alert(
    level="HIGH",
    predicted_cpu=0.85,
    current_cpu=0.53,
    action="Scale resources immediately"
)

if result:
    print("SUCCESS! Check your Slack channel!")
else:
    print("Failed. Something went wrong.")