"""
Email Delivery for Weekly Reports
Sends analytics reports via SendGrid
"""
import os
from pathlib import Path
from typing import Optional
import logging

try:
    from sendgrid import SendGridAPIClient
    from sendgrid.helpers.mail import Mail, Attachment, FileContent, FileName, FileType, Disposition
    import base64
    SENDGRID_AVAILABLE = True
except ImportError:
    SENDGRID_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmailDelivery:
    """Send weekly analytics reports via email."""
    
    def __init__(
        self,
        sendgrid_api_key: Optional[str] = None,
        from_email: Optional[str] = None,
        to_email: Optional[str] = None
    ):
        """
        Initialize email delivery.
        
        Args:
            sendgrid_api_key: SendGrid API key (or set SENDGRID_API_KEY env var)
            from_email: Sender email (or set FROM_EMAIL env var)
            to_email: Recipient email (or set TO_EMAIL env var)
        """
        if not SENDGRID_AVAILABLE:
            raise ImportError("SendGrid not installed. Run: pip install sendgrid")
        
        self.api_key = sendgrid_api_key or os.getenv("SENDGRID_API_KEY")
        self.from_email = from_email or os.getenv("FROM_EMAIL", "analytics@v7p3r.com")
        self.to_email = to_email or os.getenv("TO_EMAIL")
        
        if not self.api_key:
            raise ValueError("SendGrid API key not provided")
        if not self.to_email:
            raise ValueError("Recipient email not provided")
        
        self.client = SendGridAPIClient(self.api_key)
    
    def send_weekly_report(
        self,
        markdown_report: Path,
        json_report: Path,
        week_start: str,
        week_end: str
    ) -> bool:
        """
        Send weekly analytics report.
        
        Args:
            markdown_report: Path to markdown report
            json_report: Path to JSON report
            week_start: Start date string
            week_end: End date string
            
        Returns:
            True if sent successfully
        """
        try:
            # Read markdown content for email body
            with open(markdown_report, 'r') as f:
                markdown_content = f.read()
            
            # Convert markdown to simple HTML (basic formatting)
            html_content = self._markdown_to_simple_html(markdown_content)
            
            # Create email
            message = Mail(
                from_email=self.from_email,
                to_emails=self.to_email,
                subject=f"V7P3R Weekly Analytics Report ({week_start} - {week_end})",
                html_content=html_content
            )
            
            # Attach markdown report
            with open(markdown_report, 'rb') as f:
                markdown_data = f.read()
                encoded = base64.b64encode(markdown_data).decode()
            
            markdown_attachment = Attachment(
                FileContent(encoded),
                FileName(markdown_report.name),
                FileType('text/markdown'),
                Disposition('attachment')
            )
            message.attachment = markdown_attachment
            
            # Attach JSON report
            with open(json_report, 'rb') as f:
                json_data = f.read()
                encoded = base64.b64encode(json_data).decode()
            
            json_attachment = Attachment(
                FileContent(encoded),
                FileName(json_report.name),
                FileType('application/json'),
                Disposition('attachment')
            )
            message.add_attachment(json_attachment)
            
            # Send
            response = self.client.send(message)
            
            if response.status_code in [200, 201, 202]:
                logger.info(f"Report sent successfully to {self.to_email}")
                return True
            else:
                logger.error(f"Failed to send email: {response.status_code} - {response.body}")
                return False
                
        except Exception as e:
            logger.error(f"Error sending email: {e}", exc_info=True)
            return False
    
    def _markdown_to_simple_html(self, markdown: str) -> str:
        """Convert markdown to basic HTML for email."""
        html = markdown
        
        # Headers
        html = html.replace('# ', '<h1>').replace('\n', '</h1>\n', 1)
        html = html.replace('## ', '<h2>').replace('\n', '</h2>\n')
        html = html.replace('### ', '<h3>').replace('\n', '</h3>\n')
        
        # Bold
        import re
        html = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', html)
        
        # Lists
        html = re.sub(r'^\- (.*?)$', r'<li>\1</li>', html, flags=re.MULTILINE)
        html = html.replace('<li>', '<ul><li>', 1)
        html = html.replace('</li>\n\n', '</li></ul>\n\n')
        
        # Tables (basic support)
        lines = html.split('\n')
        in_table = False
        new_lines = []
        
        for line in lines:
            if '|' in line and '---' not in line:
                if not in_table:
                    new_lines.append('<table border="1" cellpadding="5" cellspacing="0">')
                    in_table = True
                
                cells = [cell.strip() for cell in line.split('|')[1:-1]]
                row = '<tr>' + ''.join(f'<td>{cell}</td>' for cell in cells) + '</tr>'
                new_lines.append(row)
            elif '---' in line:
                continue  # Skip separator
            else:
                if in_table:
                    new_lines.append('</table>')
                    in_table = False
                new_lines.append(line)
        
        if in_table:
            new_lines.append('</table>')
        
        html = '\n'.join(new_lines)
        
        # Wrap in basic HTML
        html = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
                h1 {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
                h2 {{ color: #34495e; margin-top: 30px; }}
                h3 {{ color: #7f8c8d; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ padding: 12px; text-align: left; border: 1px solid #ddd; }}
                th {{ background-color: #3498db; color: white; }}
                tr:nth-child(even) {{ background-color: #f2f2f2; }}
                ul {{ margin: 10px 0; padding-left: 20px; }}
                li {{ margin: 5px 0; }}
                .recommendation {{ background-color: #e8f5e9; padding: 15px; border-left: 4px solid #4caf50; margin: 20px 0; }}
            </style>
        </head>
        <body>
            {html}
        </body>
        </html>
        """
        
        return html


if __name__ == "__main__":
    # Test email delivery
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python email_delivery.py <markdown_report> <json_report>")
        print("\nEnvironment variables required:")
        print("  SENDGRID_API_KEY - Your SendGrid API key")
        print("  TO_EMAIL - Recipient email address")
        print("  FROM_EMAIL (optional) - Sender email (default: analytics@v7p3r.com)")
        sys.exit(1)
    
    markdown_path = Path(sys.argv[1])
    json_path = Path(sys.argv[2])
    
    if not markdown_path.exists() or not json_path.exists():
        print("Error: Report files not found")
        sys.exit(1)
    
    try:
        delivery = EmailDelivery()
        success = delivery.send_weekly_report(
            markdown_path,
            json_path,
            week_start="2025-11-23",
            week_end="2025-11-29"
        )
        
        if success:
            print("✓ Report sent successfully!")
        else:
            print("✗ Failed to send report")
            sys.exit(1)
            
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
