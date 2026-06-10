import React, { useState, useEffect } from 'react'
import { useRepository } from '../contexts/RepositoryContext'
import { repositoryService } from '../services/repositoryService'
import { githubService } from '../services/githubService'
import Card from '../components/common/Card'
import Button from '../components/common/Button'
import { FolderGit2, CheckCircle, XCircle, Github, GitBranch } from 'lucide-react'

const Repository = () => {
  const { repositoryPath } = useRepository()
  const [remoteUrl, setRemoteUrl] = useState('')
  const [message, setMessage] = useState('')
  const [forceUpdate, setForceUpdate] = useState(false)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  
  // GitHub 관련 상태
  const [githubRemoteName, setGithubRemoteName] = useState('origin')
  const [githubRemoteUrl, setGithubRemoteUrl] = useState('')
  const [githubRemotes, setGithubRemotes] = useState({})
  const [githubLoading, setGithubLoading] = useState(false)
  
  // GitHub 원격 저장소 목록 로드
  useEffect(() => {
    if (repositoryPath) {
      loadGithubRemotes()
    }
  }, [repositoryPath])
  
  const loadGithubRemotes = async () => {
    if (!repositoryPath) return
    try {
      const response = await githubService.getRemotes(repositoryPath)
      if (response.success && response.remotes) {
        setGithubRemotes(response.remotes)
      }
    } catch (err) {
      // 원격 저장소가 없을 수 있으므로 에러 무시
      setGithubRemotes({})
    }
  }

  const handleInit = async () => {
    if (!repositoryPath) {
      setError('저장소 경로가 설정되지 않았습니다. Home 페이지에서 경로를 설정해주세요.')
      return
    }

    setLoading(true)
    setError(null)
    setResult(null)

    try {
      const response = await repositoryService.initRepository(repositoryPath, remoteUrl || null)
      setResult(response)
      setMessage('Repository가 성공적으로 초기화되었습니다.')
    } catch (err) {
      setError(err.response?.data?.detail || err.message || 'Repository 초기화에 실패했습니다.')
    } finally {
      setLoading(false)
    }
  }

  const handleStatus = async () => {
    if (!repositoryPath) {
      setError('저장소 경로가 설정되지 않았습니다. Home 페이지에서 경로를 설정해주세요.')
      return
    }

    setLoading(true)
    setError(null)
    setResult(null)

    try {
      const response = await repositoryService.getStatus(repositoryPath)
      setResult(response)
      setMessage('상태를 성공적으로 조회했습니다.')
    } catch (err) {
      setError(err.response?.data?.detail || err.message || '상태 조회에 실패했습니다.')
    } finally {
      setLoading(false)
    }
  }

  const handleUpdate = async () => {
    if (!repositoryPath) {
      setError('저장소 경로가 설정되지 않았습니다. Home 페이지에서 경로를 설정해주세요.')
      return
    }

    setLoading(true)
    setError(null)
    setResult(null)

    try {
      const response = await repositoryService.update(repositoryPath, forceUpdate)
      setResult(response)
      setMessage('Repository가 성공적으로 업데이트되었습니다.')
    } catch (err) {
      setError(err.response?.data?.detail || err.message || '업데이트에 실패했습니다.')
    } finally {
      setLoading(false)
    }
  }
  
  const handleSetGithubRemote = async () => {
    if (!repositoryPath || !githubRemoteUrl) {
      setError('저장소 경로와 GitHub URL을 입력해주세요.')
      return
    }

    setGithubLoading(true)
    setError(null)
    setResult(null)

    try {
      const response = await githubService.setRemote(repositoryPath, githubRemoteName, githubRemoteUrl)
      setResult(response)
      setMessage('GitHub 원격 저장소가 성공적으로 설정되었습니다.')
      await loadGithubRemotes()
      setGithubRemoteUrl('')
    } catch (err) {
      setError(err.response?.data?.detail || err.message || 'GitHub 원격 저장소 설정에 실패했습니다.')
    } finally {
      setGithubLoading(false)
    }
  }
  
  const handlePushToGithub = async () => {
    if (!repositoryPath) {
      setError('저장소 경로가 설정되지 않았습니다.')
      return
    }

    setGithubLoading(true)
    setError(null)
    setResult(null)

    try {
      const response = await githubService.push(repositoryPath, githubRemoteName)
      setResult(response)
      setMessage('GitHub에 성공적으로 푸시되었습니다.')
    } catch (err) {
      setError(err.response?.data?.detail || err.message || 'GitHub 푸시에 실패했습니다.')
    } finally {
      setGithubLoading(false)
    }
  }

  return (
    <div>
      <h2 className="mb-3">Repository 관리</h2>
      
      {repositoryPath && (
        <div className="mb-3" style={{ 
          padding: '0.75rem', 
          background: 'var(--bg-secondary)', 
          borderRadius: '0.5rem',
          border: '1px solid var(--border-color)'
        }}>
          <strong>저장소 경로:</strong> <code>{repositoryPath}</code>
        </div>
      )}

      {!repositoryPath && (
        <div className="alert alert-error mb-3">
          저장소 경로가 설정되지 않았습니다. Home 페이지에서 경로를 설정해주세요.
        </div>
      )}

      <Card title="Repository 초기화">
        <div className="form-group">
          <label className="form-label">원격 저장소 URL (선택)</label>
          <input
            type="text"
            className="form-input"
            value={remoteUrl}
            onChange={(e) => setRemoteUrl(e.target.value)}
            placeholder="s3://bucket-name"
          />
        </div>
        <div className="flex gap-2">
          <Button variant="primary" onClick={handleInit} disabled={loading}>
            <FolderGit2 size={16} />
            초기화
          </Button>
          <Button variant="secondary" onClick={handleStatus} disabled={loading}>
            상태 조회
          </Button>
        </div>
        <div className="form-group mt-3">
          <label className="form-label">
            <input
              type="checkbox"
              className="form-checkbox"
              checked={forceUpdate}
              onChange={(e) => setForceUpdate(e.target.checked)}
            />
            강제 업데이트 (저장되지 않은 파일 덮어쓰기)
          </label>
        </div>
        <div className="flex gap-2 mt-2">
          <Button variant="success" onClick={handleUpdate} disabled={loading}>
            업데이트
          </Button>
        </div>
      </Card>

      {error && (
        <div className="alert alert-error">
          <XCircle size={16} />
          {error}
        </div>
      )}

      {result && result.success && (
        <div className="alert alert-success">
          <CheckCircle size={16} />
          {message || '작업이 성공적으로 완료되었습니다.'}
        </div>
      )}

      <Card title="GitHub 원격 저장소 설정">
        <div className="form-group">
          <label className="form-label">원격 저장소 이름</label>
          <input
            type="text"
            className="form-input"
            value={githubRemoteName}
            onChange={(e) => setGithubRemoteName(e.target.value)}
            placeholder="origin"
          />
        </div>
        <div className="form-group">
          <label className="form-label">GitHub 저장소 URL *</label>
          <input
            type="text"
            className="form-input"
            value={githubRemoteUrl}
            onChange={(e) => setGithubRemoteUrl(e.target.value)}
            placeholder="https://github.com/user/repo.git"
          />
        </div>
        <div className="flex gap-2">
          <Button variant="primary" onClick={handleSetGithubRemote} disabled={githubLoading || !repositoryPath}>
            <Github size={16} />
            원격 저장소 설정
          </Button>
          <Button variant="secondary" onClick={loadGithubRemotes} disabled={githubLoading || !repositoryPath}>
            목록 새로고침
          </Button>
        </div>
        
        {Object.keys(githubRemotes).length > 0 && (
          <div className="mt-3">
            <strong>설정된 원격 저장소:</strong>
            <ul style={{ marginTop: '0.5rem', paddingLeft: '1.5rem' }}>
              {Object.entries(githubRemotes).map(([name, url]) => (
                <li key={name} style={{ marginBottom: '0.5rem' }}>
                  <code>{name}</code>: <code>{url}</code>
                </li>
              ))}
            </ul>
          </div>
        )}
        
        {Object.keys(githubRemotes).length > 0 && (
          <div className="mt-3">
            <Button variant="success" onClick={handlePushToGithub} disabled={githubLoading || !repositoryPath}>
              <GitBranch size={16} />
              GitHub에 푸시
            </Button>
          </div>
        )}
      </Card>

      {result && (
        <Card title="결과">
          <pre style={{ 
            background: 'var(--bg-tertiary)', 
            padding: '1rem', 
            borderRadius: '0.5rem',
            overflow: 'auto',
            fontSize: '0.875rem'
          }}>
            {JSON.stringify(result, null, 2)}
          </pre>
        </Card>
      )}

      {(loading || githubLoading) && <div className="loading">처리 중...</div>}
    </div>
  )
}

export default Repository

