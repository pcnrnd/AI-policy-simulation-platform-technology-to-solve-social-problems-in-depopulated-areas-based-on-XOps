import React, { useState } from 'react'
import { Link } from 'react-router-dom'
import { useRepository } from '../contexts/RepositoryContext'
import { FolderGit2, GitBranch, GitCommit, Database } from 'lucide-react'
import Card from '../components/common/Card'
import Button from '../components/common/Button'

const Home = () => {
  const { repositoryPath, updateRepositoryPath } = useRepository()
  const [pathInput, setPathInput] = useState(repositoryPath || '')

  const handleSetPath = () => {
    if (pathInput.trim()) {
      updateRepositoryPath(pathInput.trim())
      alert('저장소 경로가 설정되었습니다.')
    }
  }

  return (
    <div>
      <h2 className="mb-3">DataOps Standard Editor</h2>
      <p className="mb-3" style={{ color: 'var(--text-secondary)' }}>
        데이터옵스를 위한 표준 명세 편집기/해석기
      </p>

      <Card title="저장소 경로 설정" className="mb-3">
        <div className="form-group">
          <label className="form-label">저장소 경로 *</label>
          <input
            type="text"
            className="form-input"
            value={pathInput}
            onChange={(e) => setPathInput(e.target.value)}
            placeholder="/path/to/repository"
          />
        </div>
        <Button variant="primary" onClick={handleSetPath}>
          경로 설정
        </Button>
        {repositoryPath && (
          <div className="mt-2" style={{ 
            padding: '0.75rem', 
            background: 'var(--bg-secondary)', 
            borderRadius: '0.5rem',
            border: '1px solid var(--border-color)'
          }}>
            <strong>현재 설정된 경로:</strong> <code>{repositoryPath}</code>
          </div>
        )}
      </Card>

      <div className="grid grid-3">
        <Card title="Repository" actions={<FolderGit2 size={24} />}>
          <p className="mb-2" style={{ color: 'var(--text-secondary)' }}>
            데이터 저장소를 초기화하고 관리합니다.
          </p>
          <Link to="/repository">
            <Button variant="primary">Repository 관리</Button>
          </Link>
        </Card>

        <Card title="Branch" actions={<GitBranch size={24} />}>
          <p className="mb-2" style={{ color: 'var(--text-secondary)' }}>
            브랜치를 생성하고 관리합니다.
          </p>
          <Link to="/branch">
            <Button variant="primary">Branch 관리</Button>
          </Link>
        </Card>

        <Card title="Commit" actions={<GitCommit size={24} />}>
          <p className="mb-2" style={{ color: 'var(--text-secondary)' }}>
            변경사항을 커밋하고 히스토리를 확인합니다.
          </p>
          <Link to="/commit">
            <Button variant="primary">Commit 관리</Button>
          </Link>
        </Card>

        <Card title="MinIO" actions={<Database size={24} />}>
          <p className="mb-2" style={{ color: 'var(--text-secondary)' }}>
            MinIO 버킷과 파일을 직접 관리합니다.
          </p>
          <Link to="/minio">
            <Button variant="primary">MinIO 관리</Button>
          </Link>
        </Card>
      </div>
    </div>
  )
}

export default Home

