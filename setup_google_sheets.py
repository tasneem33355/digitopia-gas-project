#!/usr/bin/env python3
"""
Setup script for Google Sheets integration
This script helps configure the shared_state system to work with Google Sheets
"""

import os
import json
import sys
from pathlib import Path

def create_streamlit_directory():
    """Create .streamlit directory if it doesn't exist"""
    streamlit_dir = Path(".streamlit")
    if not streamlit_dir.exists():
        streamlit_dir.mkdir()
        print("✅ Created .streamlit directory")
    else:
        print("ℹ️ .streamlit directory already exists")
    return streamlit_dir

def create_secrets_template():
    """Create secrets.toml template"""
    streamlit_dir = create_streamlit_directory()
    secrets_file = streamlit_dir / "secrets.toml"
    
    if secrets_file.exists():
        print("⚠️ secrets.toml already exists")
        overwrite = input("Do you want to overwrite it? (y/N): ").lower().strip()
        if overwrite != 'y':
            return False
    
    template_content = '''# Google Sheets Configuration for Digitopia Gas Project
GOOGLE_SHEETS_ID = "your_google_sheets_id_here"

[SERVICE_ACCOUNT]
type = "service_account"
project_id = "your-project-id"
private_key_id = "your-private-key-id"
private_key = """-----BEGIN PRIVATE KEY-----
your-private-key-here
-----END PRIVATE KEY-----"""
client_email = "your-service-account-email@your-project.iam.gserviceaccount.com"
client_id = "your-client-id"
auth_uri = "https://accounts.google.com/o/oauth2/auth"
token_uri = "https://oauth2.googleapis.com/token"
auth_provider_x509_cert_url = "https://www.googleapis.com/oauth2/v1/certs"
client_x509_cert_url = "https://www.googleapis.com/robot/v1/metadata/x509/your-service-account-email%40your-project.iam.gserviceaccount.com"
universe_domain = "googleapis.com"
'''
    
    with open(secrets_file, 'w') as f:
        f.write(template_content)
    
    print(f"✅ Created secrets template at {secrets_file}")
    return True

def setup_from_service_account_json():
    """Help user setup from a Google Cloud service account JSON file"""
    json_path = input("Enter path to your service account JSON file: ").strip()
    
    if not os.path.exists(json_path):
        print(f"❌ File not found: {json_path}")
        return False
    
    try:
        with open(json_path, 'r') as f:
            service_account = json.load(f)
        
        # Verify required fields
        required_fields = ['type', 'project_id', 'private_key_id', 'private_key', 
                          'client_email', 'client_id', 'auth_uri', 'token_uri', 
                          'auth_provider_x509_cert_url', 'client_x509_cert_url']
        
        missing_fields = [field for field in required_fields if field not in service_account]
        if missing_fields:
            print(f"❌ Missing fields in JSON: {missing_fields}")
            return False
        
        # Get Google Sheets ID
        sheets_id = input("Enter your Google Sheets ID (from the URL): ").strip()
        if not sheets_id:
            print("❌ Google Sheets ID is required")
            return False
        
        # Create secrets.toml
        streamlit_dir = create_streamlit_directory()
        secrets_file = streamlit_dir / "secrets.toml"
        
        secrets_content = f'''# Google Sheets Configuration - Auto-generated
GOOGLE_SHEETS_ID = "{sheets_id}"

[SERVICE_ACCOUNT]
type = "{service_account['type']}"
project_id = "{service_account['project_id']}"
private_key_id = "{service_account['private_key_id']}"
private_key = """{service_account['private_key']}"""
client_email = "{service_account['client_email']}"
client_id = "{service_account['client_id']}"
auth_uri = "{service_account['auth_uri']}"
token_uri = "{service_account['token_uri']}"
auth_provider_x509_cert_url = "{service_account['auth_provider_x509_cert_url']}"
client_x509_cert_url = "{service_account['client_x509_cert_url']}"
universe_domain = "{service_account.get('universe_domain', 'googleapis.com')}"
'''
        
        with open(secrets_file, 'w') as f:
            f.write(secrets_content)
        
        print(f"✅ Successfully created secrets.toml from JSON file")
        print(f"📁 Location: {secrets_file.absolute()}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error processing JSON file: {e}")
        return False

def test_configuration():
    """Test the Google Sheets configuration"""
    print("\n🧪 Testing Google Sheets configuration...")
    
    try:
        # Import our shared_state module
        import shared_state
        
        # Test connection
        client = shared_state.get_sheets_client()
        if client:
            print("✅ Google Sheets client connection successful")
            
            # Test worksheet access
            worksheet = shared_state.get_worksheet()
            if worksheet:
                print("✅ Worksheet access successful")
                print(f"   - Worksheet title: {worksheet.title}")
                return True
            else:
                print("❌ Failed to access worksheet")
                return False
        else:
            print("❌ Failed to connect to Google Sheets")
            return False
            
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def main():
    """Main setup function"""
    print("🚀 Google Sheets Setup for Digitopia Gas Project")
    print("=" * 50)
    
    print("\nChoose setup method:")
    print("1. Create template file (manual configuration)")
    print("2. Setup from service account JSON file")
    print("3. Test current configuration")
    print("4. Exit")
    
    choice = input("\nEnter your choice (1-4): ").strip()
    
    if choice == "1":
        if create_secrets_template():
            print("\n📋 Next steps:")
            print("1. Go to Google Cloud Console (https://console.cloud.google.com)")
            print("2. Create a new project or select existing one")
            print("3. Enable Google Sheets API and Google Drive API")
            print("4. Create a Service Account and download JSON key")
            print("5. Create a Google Sheet and share it with service account email")
            print("6. Edit .streamlit/secrets.toml with your values")
            print("7. Run this script again with option 3 to test")
    
    elif choice == "2":
        if setup_from_service_account_json():
            print("\n📋 Next steps:")
            print("1. Make sure your Google Sheet is shared with the service account email")
            print("2. Run this script again with option 3 to test")
    
    elif choice == "3":
        if test_configuration():
            print("\n🎉 Configuration is working! You can now use the dashboard and chatbot.")
        else:
            print("\n⚠️ Configuration needs fixing. Check the errors above.")
    
    elif choice == "4":
        print("👋 Goodbye!")
    
    else:
        print("❌ Invalid choice. Please run the script again.")

if __name__ == "__main__":
    main()