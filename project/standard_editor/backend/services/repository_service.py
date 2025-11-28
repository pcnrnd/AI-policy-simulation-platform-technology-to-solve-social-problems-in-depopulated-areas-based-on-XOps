"""
Repository 서비스
Repository 비즈니스 로직을 통합하는 서비스 클래스입니다.
"""
from typing import Dict, Optional

from services.dvc_service import DVCService
from services.git_service import GitService
from core.exceptions import RepositoryError, BranchError
from core.logger import logger
from config.settings import settings


class RepositoryService:
    """Repository 비즈니스 로직 통합 서비스"""
    
    def __init__(self, dvc_service: DVCService, git_service: GitService):
        """
        초기화
        
        Args:
            dvc_service: DVC 서비스 인스턴스
            git_service: Git 서비스 인스턴스
        """
        self.dvc_service = dvc_service
        self.git_service = git_service
    
    def init_repository(
        self, 
        path: str, 
        remote_url: Optional[str] = None
    ) -> Dict:
        """
        Repository 초기화 (Git + DVC)
        
        Args:
            path: 저장소 경로
            remote_url: 원격 저장소 URL (선택)
            
        Returns:
            초기화 결과
        """
        try:
            results = {}
            
            # Git 초기화
            logger.info(f"Initializing Git repository at {path}")
            git_result = self.git_service.init(path)
            results["git"] = git_result
            
            # DVC 초기화
            logger.info(f"Initializing DVC repository at {path}")
            dvc_result = self.dvc_service.init(path)
            results["dvc"] = dvc_result
            
            # 원격 저장소 설정 (제공된 경우)
            if remote_url:
                logger.info(f"Setting up DVC remote: {remote_url}")
                remote_result = self.dvc_service.remote_add(
                    name=settings.dvc_remote_name,
                    url=remote_url or settings.dvc_remote_url,
                    path=path,
                    endpoint_url=settings.minio_endpoint,
                    access_key_id=settings.minio_access_key,
                    secret_access_key=settings.minio_secret_key
                )
                results["remote"] = remote_result
            
            return {
                "success": True,
                "message": "Repository initialized successfully",
                "results": results
            }
        except Exception as e:
            logger.error(f"Repository initialization failed: {str(e)}")
            raise RepositoryError(f"Repository initialization failed: {str(e)}")
    
    def checkout(self, branch: str, path: str, create: bool = False) -> Dict:
        """
        브랜치 체크아웃
        
        Args:
            branch: 브랜치 이름
            path: 저장소 경로
            create: 새 브랜치 생성 여부
            
        Returns:
            체크아웃 결과
        """
        try:
            logger.info(f"Checking out branch: {branch} at {path}")
            result = self.git_service.checkout(branch, path, create=create)
            
            return {
                "success": True,
                "message": f"Checked out branch: {branch}",
                "data": result
            }
        except Exception as e:
            logger.error(f"Checkout failed: {str(e)}")
            raise BranchError(f"Checkout failed: {str(e)}")
    
    def commit(
        self, 
        message: str, 
        path: str,
        push: bool = False
    ) -> Dict:
        """
        변경사항 커밋 (Git + DVC)
        
        Args:
            message: 커밋 메시지
            path: 저장소 경로
            push: 원격 저장소로 푸시 여부
            
        Returns:
            커밋 결과
        """
        try:
            results = {}
            
            # DVC 커밋
            logger.info(f"Committing DVC changes: {message}")
            dvc_result = self.dvc_service.commit(message, path)
            results["dvc"] = dvc_result
            
            # Git 커밋
            logger.info(f"Committing Git changes: {message}")
            git_result = self.git_service.commit(message, path)
            results["git"] = git_result
            
            # 푸시 (요청된 경우)
            if push:
                logger.info("Pushing to remote")
                dvc_push = self.dvc_service.push(path)
                git_push = self.git_service.push(path=path)
                results["push"] = {"dvc": dvc_push, "git": git_push}
            
            return {
                "success": True,
                "message": "Changes committed successfully",
                "results": results
            }
        except Exception as e:
            logger.error(f"Commit failed: {str(e)}")
            raise RepositoryError(f"Commit failed: {str(e)}")
    
    def add_files(self, files: list, path: str) -> Dict:
        """
        파일 추가 (Git + DVC)
        
        Args:
            files: 추가할 파일 경로 리스트
            path: 저장소 경로
            
        Returns:
            추가 결과
        """
        try:
            results = {}
            
            # DVC에 추가
            logger.info(f"Adding files to DVC: {files}")
            dvc_result = self.dvc_service.add(files, path)
            results["dvc"] = dvc_result
            
            # Git에 추가
            logger.info(f"Adding files to Git: {files}")
            git_result = self.git_service.add(files, path)
            results["git"] = git_result
            
            return {
                "success": True,
                "message": "Files added successfully",
                "results": results
            }
        except Exception as e:
            logger.error(f"Add files failed: {str(e)}")
            raise RepositoryError(f"Add files failed: {str(e)}")
    
    def update(self, path: str) -> Dict:
        """
        원격 저장소에서 업데이트 (Pull)
        
        Args:
            path: 저장소 경로
            
        Returns:
            업데이트 결과
        """
        try:
            results = {}
            
            # DVC 풀
            logger.info(f"Pulling DVC changes at {path}")
            dvc_result = self.dvc_service.pull(path)
            results["dvc"] = dvc_result
            
            # Git 풀
            logger.info(f"Pulling Git changes at {path}")
            git_result = self.git_service.pull(path=path)
            results["git"] = git_result
            
            return {
                "success": True,
                "message": "Repository updated successfully",
                "results": results
            }
        except Exception as e:
            logger.error(f"Update failed: {str(e)}")
            raise RepositoryError(f"Update failed: {str(e)}")
    
    def merge(
        self, 
        source_branch: str, 
        target_branch: str, 
        path: str
    ) -> Dict:
        """
        브랜치 병합
        
        Args:
            source_branch: 병합할 소스 브랜치
            target_branch: 타겟 브랜치
            path: 저장소 경로
            
        Returns:
            병합 결과
        """
        try:
            # 타겟 브랜치로 체크아웃
            logger.info(f"Checking out target branch: {target_branch}")
            self.git_service.checkout(target_branch, path)
            
            # 소스 브랜치 병합
            logger.info(f"Merging {source_branch} into {target_branch}")
            result = self.git_service.merge(source_branch, path)
            
            return {
                "success": True,
                "message": f"Merged {source_branch} into {target_branch}",
                "data": result
            }
        except Exception as e:
            logger.error(f"Merge failed: {str(e)}")
            raise BranchError(f"Merge failed: {str(e)}")
    
    def get_branches(self, path: str) -> Dict:
        """
        브랜치 목록 조회
        
        Args:
            path: 저장소 경로
            
        Returns:
            브랜치 목록
        """
        try:
            result = self.git_service.branch_list(path)
            return {
                "success": True,
                "data": result
            }
        except Exception as e:
            logger.error(f"Get branches failed: {str(e)}")
            raise BranchError(f"Get branches failed: {str(e)}")
    
    def get_status(self, path: str) -> Dict:
        """
        저장소 상태 조회
        
        Args:
            path: 저장소 경로
            
        Returns:
            저장소 상태
        """
        try:
            dvc_status = self.dvc_service.status(path)
            return {
                "success": True,
                "data": {
                    "dvc": dvc_status
                }
            }
        except Exception as e:
            logger.error(f"Get status failed: {str(e)}")
            raise RepositoryError(f"Get status failed: {str(e)}")

