import { BrowserRouter as Router, Routes, Route } from 'react-router-dom'
import { RepositoryProvider } from './contexts/RepositoryContext'
import Home from './pages/Home'
import Repository from './pages/Repository'
import Branch from './pages/Branch'
import Commit from './pages/Commit'
import MinIO from './pages/MinIO'
import './styles/main.css'
import './styles/components.css'

function App() {
  return (
    <RepositoryProvider>
      <Router>
        <div className="app">
          <header className="app-header">
            <h1>DataOps Standard Editor</h1>
          </header>
          <main className="app-main">
            <Routes>
              <Route path="/" element={<Home />} />
              <Route path="/repository" element={<Repository />} />
              <Route path="/branch" element={<Branch />} />
              <Route path="/commit" element={<Commit />} />
              <Route path="/minio" element={<MinIO />} />
            </Routes>
          </main>
        </div>
      </Router>
    </RepositoryProvider>
  )
}

export default App

