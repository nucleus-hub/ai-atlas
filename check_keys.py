import os
from dotenv import load_dotenv

load_dotenv()  # Reads .env from the current folder.

for variable_name in ("GOOGLE_API_KEY", "HF_TOKEN"):
	value = os.environ.get(variable_name)
	if not value:
		raise SystemExit(
			f"{variable_name} is not set. Add it to your .env file."
		)
	print(f"{variable_name}=...{value[-4:]}")
