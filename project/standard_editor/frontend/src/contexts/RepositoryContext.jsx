import React, { createContext, useContext, useState } from 'react'

const RepositoryContext = createContext()

export const useRepository = () => {
  const context = useContext(RepositoryContext)
  if (!context) {
    throw new Error('useRepository must be used within a RepositoryProvider')
  }
  return context
}

export const RepositoryProvider = ({ children }) => {
  const [repositoryPath, setRepositoryPath] = useState('')

  const updateRepositoryPath = (path) => {
    setRepositoryPath(path)
  }

  return (
    <RepositoryContext.Provider
      value={{
        repositoryPath,
        updateRepositoryPath,
      }}
    >
      {children}
    </RepositoryContext.Provider>
  )
}

