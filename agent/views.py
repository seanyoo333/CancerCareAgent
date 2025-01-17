from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.http import FileResponse
from agent.crew import crew  # Import the crew instance
import logging

logger = logging.getLogger(__name__)


class ChatAPIView(APIView):
    def post(self, request):
        try:
            message = request.data.get('message')
            response_type = request.data.get('response_type', 'text')
            prescription_needed = request.data.get('prescription', 'No')

            if not message:
                return Response({
                    'status': 'error',
                    'message': 'Message is required'
                }, status=status.HTTP_400_BAD_REQUEST)

            # Process message using CrewAI
            result = crew.kickoff(
                inputs=dict(
                    question=message,
                    prescription=prescription_needed
                )
            )

            if response_type == 'pdf':
                # PDF 생성 로직이 필요한 경우 여기에 구현
                return FileResponse(
                    result,
                    as_attachment=True,
                    filename='response.pdf',
                    content_type='application/pdf'
                )

            return Response({
                'status': 'success',
                'type': 'text',
                'message': result
            })

        except Exception as e:
            return self._handle_error(e)

    def _handle_error(self, error):
        logger.error(f"Error in ChatAPIView: {str(error)}")
        return Response({
            'status': 'error',
            'message': str(error)
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)