from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.http import FileResponse
from agent.crew import get_crew  # crew 대신 get_crew 임포트
import logging
from rest_framework.decorators import api_view
from django.shortcuts import render

# 앱 이름으로 로거 설정
logger = logging.getLogger('agent.views')  # 'agent'는 앱 이름

def agent_test_page(request):
    """테스트 페이지를 렌더링하는 뷰"""
    return render(request, 'agent/agent_test.html')

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

            # crew 인스턴스 가져오기
            crew = get_crew()
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


@api_view(['POST'])
def health_query(request):
    """건강 관련 질문을 처리하는 API 엔드포인트"""
    try:
        question = request.data.get('question')
        prescription = request.data.get('prescription', 'No')  # 추가
        logger.info(f"수신된 건강 질문: {question}, 처방전 필요: {prescription}")
        
        if not question:
            logger.warning("빈 질문이 전송됨")
            return Response(
                {"error": "질문이 제공되지 않았습니다."}, 
                status=status.HTTP_400_BAD_REQUEST
            )

        # crew 인스턴스 가져오기
        crew = get_crew()
        logger.info("Crew에 질문 전달 시작")
        try:
            response = crew.kickoff(inputs={
                "question": question,
                "prescription": prescription  # prescription 값 추가
            })
            logger.info(f"Crew 응답 완료: {response}")
        except Exception as crew_error:
            logger.error(f"Crew 처리 중 오류: {str(crew_error)}")
            raise crew_error

        return Response({
            "status": "success",
            "message": response
        })

        #return Response({
         #   "response": response,
          #  "status": "success"
        #})

    except Exception as e:
        logger.error(f"Error in health_query: {str(e)}", exc_info=True)
        return Response(
            {"error": f"처리 중 오류가 발생했습니다: {str(e)}"}, 
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )
    
