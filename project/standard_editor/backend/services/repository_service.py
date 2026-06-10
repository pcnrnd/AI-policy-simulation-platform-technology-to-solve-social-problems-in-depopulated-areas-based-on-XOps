"""
Repository 서비스
Repository 비즈니스 로직을 통합하는 서비스 클래스입니다.
"""
from typing import Dict, Optional

from services.dvc_service import DVCService
from services.git_service import GitService
from core.exceptions import RepositoryError, BranchError, ValidationError
from core.logger import logger
from core.path_validator import validate_path, ensure_directory_exists, check_repository_initialized
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
            # 디렉토리 생성 (존재하지 않는 경우)
            repo_path = ensure_directory_exists(path)
            path_str = str(repo_path)
            
            # 이미 초기화된 저장소 확인
            init_status = check_repository_initialized(path_str)
            results = {"initialization_status": init_status}
            
            # Git 초기화 (이미 초기화되지 않은 경우)
            if not init_status["git_initialized"]:
                logger.info(f"Initializing Git repository at {path_str}")
                git_result = self.git_service.init(path_str)
                results["git"] = git_result
            else:
                logger.info(f"Git repository already initialized at {path_str}")
                results["git"] = {"message": "Already initialized", "skipped": True}
            
            # DVC 초기화 (이미 초기화되지 않은 경우)
            if not init_status["dvc_initialized"]:
                logger.info(f"Initializing DVC repository at {path_str}")
                dvc_result = self.dvc_service.init(path_str)
                results["dvc"] = dvc_result
            else:
                logger.info(f"DVC repository already initialized at {path_str}")
                results["dvc"] = {"message": "Already initialized", "skipped": True}
            
            # 원격 저장소 설정 (제공된 경우)
            if remote_url:
                logger.info(f"Setting up DVC remote: {remote_url}")
                remote_result = self.dvc_service.remote_add(
                    name=settings.dvc_remote_name,
                    url=remote_url or settings.dvc_remote_url,
                    path=path_str,
                    endpoint_url=settings.minio_endpoint,
                    access_key_id=settings.minio_access_key,
                    secret_access_key=settings.minio_secret_key
                )
                results["remote"] = remote_result
            
            message = "Repository initialized successfully"
            if init_status["git_initialized"] or init_status["dvc_initialized"]:
                message += " (some components were already initialized)"
            
            return {
                "success": True,
                "message": message,
                "results": results
            }
        except ValidationError as e:
            logger.error(f"Path validation failed: {str(e)}")
            raise RepositoryError(f"경로 검증 실패: {str(e)}")
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
            # 경로 검증
            repo_path = validate_path(path, must_exist=True, must_be_dir=True)
            path_str = str(repo_path)
            
            logger.info(f"Checking out branch: {branch} at {path_str}")
            result = self.git_service.checkout(branch, path_str, create=create)
            
            return {
                "success": True,
                "message": f"Checked out branch: {branch}",
                "data": result
            }
        except ValidationError as e:
            logger.error(f"Path validation failed: {str(e)}")
            raise BranchError(f"경로 검증 실패: {str(e)}")
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
            # 경로 검증
            repo_path = validate_path(path, must_exist=True, must_be_dir=True)
            path_str = str(repo_path)
            
            results = {}
            
            # DVC 커밋
            logger.info(f"Committing DVC changes: {message}")
            dvc_result = self.dvc_service.commit(message, path_str)
            results["dvc"] = dvc_result
            
            # Git 커밋 (변경사항 확인 포함)
            logger.info(f"Committing Git changes: {message}")
            try:
                git_result = self.git_service.commit(message, path_str, allow_empty=False)
                results["git"] = git_result
            except RepositoryError as e:
                if "커밋할 변경사항이 없습니다" in str(e):
                    # 변경사항이 없는 경우에도 성공으로 처리하되 메시지 표시
                    results["git"] = {
                        "message": "커밋할 변경사항이 없습니다.",
                        "skipped": True
                    }
                    logger.info("No changes to commit")
                else:
                    raise
            
            # 푸시 (요청된 경우)
            if push:
                logger.info("Pushing to remote")
                dvc_push = self.dvc_service.push(path_str)
                git_push = self.git_service.push(path=path_str)
                results["push"] = {"dvc": dvc_push, "git": git_push}
            
            return {
                "success": True,
                "message": "Changes committed successfully",
                "results": results
            }
        except ValidationError as e:
            logger.error(f"Path validation failed: {str(e)}")
            raise RepositoryError(f"경로 검증 실패: {str(e)}")
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
            # 경로 검증
            repo_path = validate_path(path, must_exist=True, must_be_dir=True)
            path_str = str(repo_path)
            
            results = {}
            
            # DVC에 추가
            logger.info(f"Adding files to DVC: {files}")
            dvc_result = self.dvc_service.add(files, path_str)
            results["dvc"] = dvc_result
            
            # Git에 추가
            logger.info(f"Adding files to Git: {files}")
            git_result = self.git_service.add(files, path_str)
            results["git"] = git_result
            
            return {
                "success": True,
                "message": "Files added successfully",
                "results": results
            }
        except ValidationError as e:
            logger.error(f"Path validation failed: {str(e)}")
            raise RepositoryError(f"경로 검증 실패: {str(e)}")
        except Exception as e:
            logger.error(f"Add files failed: {str(e)}")
            raise RepositoryError(f"Add files failed: {str(e)}")
    
    def update(self, path: str, force: bool = False) -> Dict:
        """
        원격 저장소에서 업데이트 (Pull)
        
        Args:
            path: 저장소 경로
            force: 강제 업데이트 여부 (저장되지 않은 파일 덮어쓰기)
            
        Returns:
            업데이트 결과
        """
        try:
            # 경로 검증
            repo_path = validate_path(path, must_exist=True, must_be_dir=True)
            path_str = str(repo_path)
            
            results = {}
            
            # DVC 풀
            logger.info(f"Pulling DVC changes at {path_str} (force={force})")
            dvc_result = self.dvc_service.pull(path_str, force=force)
            results["dvc"] = dvc_result
            
            # Git 풀
            logger.info(f"Pulling Git changes at {path_str}")
            git_result = self.git_service.pull(path=path_str)
            results["git"] = git_result
            
            return {
                "success": True,
                "message": "Repository updated successfully",
                "results": results
            }
        except ValidationError as e:
            logger.error(f"Path validation failed: {str(e)}")
            raise RepositoryError(f"경로 검증 실패: {str(e)}")
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
            # 경로 검증
            repo_path = validate_path(path, must_exist=True, must_be_dir=True)
            path_str = str(repo_path)
            
            # 타겟 브랜치로 체크아웃
            logger.info(f"Checking out target branch: {target_branch}")
            self.git_service.checkout(target_branch, path_str)
            
            # 소스 브랜치 병합
            logger.info(f"Merging {source_branch} into {target_branch}")
            result = self.git_service.merge(source_branch, path_str)
            
            return {
                "success": True,
                "message": f"Merged {source_branch} into {target_branch}",
                "data": result
            }
        except ValidationError as e:
            logger.error(f"Path validation failed: {str(e)}")
            raise BranchError(f"경로 검증 실패: {str(e)}")
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
            # 경로 검증
            repo_path = validate_path(path, must_exist=True, must_be_dir=True)
            path_str = str(repo_path)
            
            result = self.git_service.branch_list(path_str)
            return {
                "success": True,
                "data": result
            }
        except ValidationError as e:
            logger.error(f"Path validation failed: {str(e)}")
            raise BranchError(f"경로 검증 실패: {str(e)}")
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
            # 경로 검증
            repo_path = validate_path(path, must_exist=True, must_be_dir=True)
            path_str = str(repo_path)
            
            dvc_status = self.dvc_service.status(path_str)
            return {
                "success": True,
                "data": {
                    "dvc": dvc_status
                }
            }
        except ValidationError as e:
            logger.error(f"Path validation failed: {str(e)}")
            raise RepositoryError(f"경로 검증 실패: {str(e)}")
        except Exception as e:
            logger.error(f"Get status failed: {str(e)}")
            raise RepositoryError(f"Get status failed: {str(e)}")
    
    def set_github_remote(self, path: str, remote_name: str = "origin", remote_url: str = None) -> Dict:
        """
        GitHub 원격 저장소 설정
        
        Args:
            path: 저장소 경로
            remote_name: 원격 저장소 이름 (기본값: origin)
            remote_url: GitHub 저장소 URL
            
        Returns:
            설정 결과
        """
        try:
            # 경로 검증
            repo_path = validate_path(path, must_exist=True, must_be_dir=True)
            path_str = str(repo_path)
            
            if not remote_url:
                raise ValidationError("GitHub URL이 필요합니다.")
            
            # 원격 저장소 목록 확인
            try:
                remote_list = self.git_service.remote_list(path_str)
                existing_remotes = remote_list.get("remotes", {})
                
                if remote_name in existing_remotes:
                    # 이미 존재하면 URL 업데이트
                    self.git_service.remote_set_url(remote_name, remote_url, path_str)
                    logger.info(f"Remote {remote_name} URL updated to {remote_url}")
                    return {
                        "success": True,
                        "message": f"원격 저장소 '{remote_name}'의 URL이 업데이트되었습니다.",
                        "remotes": {remote_name: remote_url}
                    }
                else:
                    # 새로 추가
                    self.git_service.remote_add(remote_name, remote_url, path_str)
                    logger.info(f"Remote {remote_name} added: {remote_url}")
                    return {
                        "success": True,
                        "message": f"원격 저장소 '{remote_name}'이 추가되었습니다.",
                        "remotes": {remote_name: remote_url}
                    }
            except RepositoryError as e:
                # 원격 저장소가 없는 경우 새로 추가
                self.git_service.remote_add(remote_name, remote_url, path_str)
                logger.info(f"Remote {remote_name} added: {remote_url}")
                return {
                    "success": True,
                    "message": f"원격 저장소 '{remote_name}'이 추가되었습니다.",
                    "remotes": {remote_name: remote_url}
                }
        except ValidationError as e:
            logger.error(f"Path validation failed: {str(e)}")
            raise RepositoryError(f"경로 검증 실패: {str(e)}")
        except Exception as e:
            logger.error(f"Set GitHub remote failed: {str(e)}")
            raise RepositoryError(f"GitHub 원격 저장소 설정 실패: {str(e)}")
    
    def get_github_remotes(self, path: str) -> Dict:
        """
        GitHub 원격 저장소 목록 조회
        
        Args:
            path: 저장소 경로
            
        Returns:
            원격 저장소 목록
        """
        try:
            # 경로 검증
            repo_path = validate_path(path, must_exist=True, must_be_dir=True)
            path_str = str(repo_path)
            
            remote_list = self.git_service.remote_list(path_str)
            return {
                "success": True,
                "remotes": remote_list.get("remotes", {}),
                "data": remote_list
            }
        except ValidationError as e:
            logger.error(f"Path validation failed: {str(e)}")
            raise RepositoryError(f"경로 검증 실패: {str(e)}")
        except Exception as e:
            logger.error(f"Get GitHub remotes failed: {str(e)}")
            raise RepositoryError(f"원격 저장소 목록 조회 실패: {str(e)}")
    
    def push_to_github(self, path: str, remote_name: str = "origin", branch: Optional[str] = None) -> Dict:
        """
        GitHub에 푸시
        
        Args:
            path: 저장소 경로
            remote_name: 원격 저장소 이름 (기본값: origin)
            branch: 푸시할 브랜치 (기본값: 현재 브랜치)
            
        Returns:
            푸시 결과
        """
        try:
            # 경로 검증
            repo_path = validate_path(path, must_exist=True, must_be_dir=True)
            path_str = str(repo_path)
            
            # 원격 저장소 확인
            remote_list = self.git_service.remote_list(path_str)
            if remote_name not in remote_list.get("remotes", {}):
                raise RepositoryError(f"원격 저장소 '{remote_name}'이 설정되지 않았습니다.")
            
            # 푸시 실행
            push_result = self.git_service.push(remote_name, branch, path_str)
            
            logger.info(f"Pushed to {remote_name}/{branch or 'current branch'}")
            return {
                "success": True,
                "message": f"GitHub에 푸시되었습니다: {remote_name}/{branch or 'current branch'}",
                "data": push_result
            }
        except ValidationError as e:
            logger.error(f"Path validation failed: {str(e)}")
            raise RepositoryError(f"경로 검증 실패: {str(e)}")
        except Exception as e:
            logger.error(f"Push to GitHub failed: {str(e)}")
            raise RepositoryError(f"GitHub 푸시 실패: {str(e)}")

