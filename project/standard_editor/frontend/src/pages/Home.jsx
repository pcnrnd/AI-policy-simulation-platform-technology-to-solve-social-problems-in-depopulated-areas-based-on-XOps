import React from 'react'
import { Link } from 'react-router-dom'
import { FolderGit2, GitBranch, GitCommit } from 'lucide-react'
import Card from '../components/common/Card'
import Button from '../components/common/Button'

const Home = () => {
  return (
    <div>
      <h2 className="mb-3">DataOps Standard Editor</h2>
      <p className="mb-3" style={{ color: 'var(--text-secondary)' }}>
        데이터옵스를 위한 표준 명세 편집기/해석기
      </p>

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
      </div>
    </div>
  )
}

export default Home

