import React, { useState } from 'react'
import { repositoryService } from '../services/repositoryService'
import Card from '../components/common/Card'
import Button from '../components/common/Button'
import { GitCommit, CheckCircle, XCircle } from 'lucide-react'

const Commit = () => {
  const [path, setPath] = useState('')
  const [message, setMessage] = useState('')
  const [files, setFiles] = useState('')
  const [push, setPush] = useState(false)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)

  const handleCommit = async () => {
    if (!path || !message) {
      setError('경로와 커밋 메시지를 입력해주세요.')
      return
    }

    setLoading(true)
    setError(null)
    setResult(null)

    try {
      const response = await repositoryService.commit(message, path, push)
      setResult(response)
      setError(null)
      setMessage('')
    } catch (err) {
      setError(err.response?.data?.detail || err.message || '커밋에 실패했습니다.')
    } finally {
      setLoading(false)
    }
  }

  const handleAddFiles = async () => {
    if (!path || !files) {
      setError('경로와 파일 목록을 입력해주세요.')
      return
    }

    const fileList = files.split(',').map(f => f.trim()).filter(f => f)
    if (fileList.length === 0) {
      setError('파일을 입력해주세요.')
      return
    }

    setLoading(true)
    setError(null)
    setResult(null)

    try {
      const response = await repositoryService.addFiles(fileList, path)
      setResult(response)
      setError(null)
      setFiles('')
    } catch (err) {
      setError(err.response?.data?.detail || err.message || '파일 추가에 실패했습니다.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div>
      <h2 className="mb-3">Commit 관리</h2>

      <Card title="파일 추가">
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
          <label className="form-label">파일 경로 (쉼표로 구분) *</label>
          <input
            type="text"
            className="form-input"
            value={files}
            onChange={(e) => setFiles(e.target.value)}
            placeholder="file1.txt, file2.txt, data/file3.csv"
          />
        </div>
        <Button variant="primary" onClick={handleAddFiles} disabled={loading}>
          파일 추가
        </Button>
      </Card>

      <Card title="변경사항 커밋">
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
          <label className="form-label">커밋 메시지 *</label>
          <textarea
            className="form-input form-textarea"
            value={message}
            onChange={(e) => setMessage(e.target.value)}
            placeholder="커밋 메시지를 입력하세요..."
          />
        </div>
        <div className="form-group">
          <label className="form-label">
            <input
              type="checkbox"
              className="form-checkbox"
              checked={push}
              onChange={(e) => setPush(e.target.checked)}
            />
            원격 저장소로 푸시
          </label>
        </div>
        <Button variant="primary" onClick={handleCommit} disabled={loading}>
          <GitCommit size={16} />
          커밋
        </Button>
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
          작업이 성공적으로 완료되었습니다.
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

export default Commit

