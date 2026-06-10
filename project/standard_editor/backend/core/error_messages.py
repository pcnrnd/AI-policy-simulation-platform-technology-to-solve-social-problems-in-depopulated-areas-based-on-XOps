"""
사용자 친화적 에러 메시지 매핑
기술적 에러 메시지를 사용자가 이해하기 쉬운 메시지로 변환합니다.
"""
from typing import Optional


# 에러 메시지 매핑 딕셔너리
ERROR_MESSAGE_MAP = {
    # 경로 관련
    "경로가 존재하지 않습니다": "지정한 경로를 찾을 수 없습니다. 경로를 확인해주세요.",
    "경로가 디렉토리가 아닙니다": "지정한 경로는 파일입니다. 디렉토리 경로를 입력해주세요.",
    "경로가 파일이 아닙니다": "지정한 경로는 디렉토리입니다. 파일 경로를 입력해주세요.",
    "경로를 정규화할 수 없습니다": "입력한 경로 형식이 올바르지 않습니다. 경로를 확인해주세요.",
    "경로가 제공되지 않았습니다": "저장소 경로를 입력해주세요.",
    "디렉토리를 생성할 수 없습니다": "디렉토리를 생성할 수 없습니다. 권한을 확인해주세요.",
    
    # Git 관련
    "Git init failed": "Git 저장소 초기화에 실패했습니다.",
    "Git commit failed": "커밋에 실패했습니다.",
    "Git checkout failed": "브랜치 체크아웃에 실패했습니다.",
    "Git add failed": "파일 추가에 실패했습니다.",
    "Git merge failed": "브랜치 병합에 실패했습니다.",
    "Git push failed": "원격 저장소로 푸시에 실패했습니다.",
    "Git pull failed": "원격 저장소에서 가져오기에 실패했습니다.",
    "커밋할 변경사항이 없습니다": "커밋할 변경사항이 없습니다.",
    
    # DVC 관련
    "DVC init failed": "DVC 저장소 초기화에 실패했습니다.",
    "DVC commit failed": "DVC 커밋에 실패했습니다.",
    "DVC add failed": "DVC에 파일 추가에 실패했습니다.",
    "DVC push failed": "DVC 원격 저장소로 푸시에 실패했습니다.",
    "DVC pull failed": "DVC 원격 저장소에서 가져오기에 실패했습니다. 저장되지 않은 파일이 있으면 강제 업데이트 옵션을 사용해주세요.",
    "DVC status failed": "DVC 상태 확인에 실패했습니다.",
    "DVC remote add failed": "DVC 원격 저장소 추가에 실패했습니다.",
    
    # Repository 관련
    "Repository initialization failed": "저장소 초기화에 실패했습니다.",
    "Repository updated successfully": "저장소가 성공적으로 업데이트되었습니다.",
    "Update failed": "저장소 업데이트에 실패했습니다.",
    "Commit failed": "커밋에 실패했습니다.",
    "Add files failed": "파일 추가에 실패했습니다.",
    "Checkout failed": "체크아웃에 실패했습니다.",
    "Merge failed": "병합에 실패했습니다.",
    "Get branches failed": "브랜치 목록 조회에 실패했습니다.",
    "Get status failed": "상태 조회에 실패했습니다.",
    
    # 명령어 실행 관련
    "Command timeout": "명령어 실행 시간이 초과되었습니다.",
    "Command failed": "명령어 실행에 실패했습니다.",
    
    # 일반적인 메시지
    "경로 검증 실패": "입력한 경로가 올바르지 않습니다.",
}


def get_user_friendly_message(error_message: str) -> str:
    """
    기술적 에러 메시지를 사용자 친화적 메시지로 변환
    
    Args:
        error_message: 원본 에러 메시지
        
    Returns:
        사용자 친화적 메시지
    """
    # 정확한 매칭 시도
    if error_message in ERROR_MESSAGE_MAP:
        return ERROR_MESSAGE_MAP[error_message]
    
    # 부분 매칭 시도
    for key, value in ERROR_MESSAGE_MAP.items():
        if key.lower() in error_message.lower():
            return value
    
    # 매칭되지 않으면 원본 메시지 반환 (일부 정리)
    # 너무 긴 메시지는 요약
    if len(error_message) > 200:
        return f"오류가 발생했습니다: {error_message[:150]}..."
    
    return error_message


def format_error_detail(error: Exception, include_technical: bool = False) -> dict:
    """
    에러를 사용자 친화적인 형식으로 포맷팅
    
    Args:
        error: 예외 객체
        include_technical: 기술적 세부사항 포함 여부
        
    Returns:
        포맷팅된 에러 딕셔너리
    """
    error_message = str(error)
    user_message = get_user_friendly_message(error_message)
    
    result = {
        "message": user_message
    }
    
    if include_technical:
        result["technical_detail"] = error_message
        result["error_type"] = type(error).__name__
    
    return result

