# API Setup Guide

## Setting up Perplexity AI API

To use the question generation functionality, you need to configure the Perplexity AI API key.

### Step 1: Get Your API Key

1. Visit [Perplexity AI Settings](https://www.perplexity.ai/settings/api)
2. Sign up or log in to your account
3. Navigate to the API section
4. Generate a new API key
5. Copy the API key (it will look like: `pplx-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx`)

### Step 2: Configure the Environment

1. Copy the `.env.example` file to `.env`:
   ```bash
   cp .env.example .env
   ```

2. Open the `.env` file and replace `your_perplexity_api_key_here` with your actual API key:
   ```
   API=pplx-your-actual-api-key-here
   ```

3. Also set a secure secret key for Flask:
   ```
   SECRET_KEY=your-very-secure-random-secret-key-here
   ```

### Step 3: Install Dependencies

Install the required packages:
```bash
pip install -r requirements.txt
```

### Step 4: Run the Application

```bash
python run.py
```

## Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `API` | Perplexity AI API key | Yes |
| `SECRET_KEY` | Flask secret key for sessions | Yes |
| `FLASK_ENV` | Flask environment (development/production) | No |
| `DATABASE_URL` | Database connection string | No |

## Troubleshooting

### API Key Issues
- Make sure your API key starts with `pplx-`
- Verify you have credits/quota remaining in your Perplexity account
- Check that your internet connection is working

### Environment Variable Issues
- Ensure the `.env` file is in the project root directory
- Make sure there are no spaces around the `=` in variable assignments
- Restart the application after changing environment variables

### Import Errors
- Run `pip install -r requirements.txt` to install all dependencies
- Make sure you're using the correct Python environment/virtualenv
