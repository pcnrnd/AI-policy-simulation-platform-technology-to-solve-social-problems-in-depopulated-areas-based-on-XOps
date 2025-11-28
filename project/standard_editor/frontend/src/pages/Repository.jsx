import React, { useState } from 'react'
import { repositoryService } from '../services/repositoryService'
import Card from '../components/common/Card'
import Button from '../components/common/Button'
import { FolderGit2, CheckCircle, XCircle } from 'lucide-react'

const Repository = () => {
  const [path, setPath] = useState('')
  const [remoteUrl, setRemoteUrl] = useState('')
  const [message, setMessage] = useState('')
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)

  const handleInit = async () => {
    if (!path) {
      setError('경로를 입력해주세요.')
      return
    }

    setLoading(true)
    setError(null)
    setResult(null)

    try {
      const response = await repositoryService.initRepository(path, remoteUrl || null)
      setResult(response)
      setMessage('Repository가 성공적으로 초기화되었습니다.')
    } catch (err) {
      setError(err.response?.data?.detail || err.message || 'Repository 초기화에 실패했습니다.')
    } finally {
      setLoading(false)
    }
  }

  const handleStatus = async () => {
    if (!path) {
      setError('경로를 입력해주세요.')
      return
    }

    setLoading(true)
    setError(null)
    setResult(null)

    try {
      const response = await repositoryService.getStatus(path)
      setResult(response)
      setMessage('상태를 성공적으로 조회했습니다.')
    } catch (err) {
      setError(err.response?.data?.detail || err.message || '상태 조회에 실패했습니다.')
    } finally {
      setLoading(false)
    }
  }

  const handleUpdate = async () => {
    if (!path) {
      setError('경로를 입력해주세요.')
      return
    }

    setLoading(true)
    setError(null)
    setResult(null)

    try {
      const response = await repositoryService.update(path)
      setResult(response)
      setMessage('Repository가 성공적으로 업데이트되었습니다.')
    } catch (err) {
      setError(err.response?.data?.detail || err.message || '업데이트에 실패했습니다.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div>
      <h2 className="mb-3">Repository 관리</h2>

      <Card title="Repository 초기화">
        <div className="form-group">
          <label className="form-label">저장소 경로 *</label>
          <input
            type="text"
            className="form-input"
            value={path}
            onChange={(e) => setPath(e.target.value)}
            placeholder="/path/to/repository"
          />
        </div>
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

      {loading && <div className="loading">처리 중...</div>}
    </div>
  )
}

export default Repository

