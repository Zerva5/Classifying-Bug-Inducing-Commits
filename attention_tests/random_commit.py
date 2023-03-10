import git
import random

def get_random_commit_shas(directory, n):
    # Open the Git repository
    repo = git.Repo(directory)

    # Get the SHA of the latest commit on the main branch
    latest_commit_sha = repo.head.commit.hexsha

    # Get the SHA of the first commit on the main branch
    first_commit_sha = repo.git.rev_list('--max-parents=0', 'main').splitlines()[0]

    # Create a list of all the commit SHAs on the main branch
    all_commit_shas = [commit.hexsha for commit in repo.iter_commits('main')]

    # Exclude the most recent and first ever commits from the list of possible random commit SHAs
    possible_commit_shas = [sha for sha in all_commit_shas if sha != latest_commit_sha and sha != first_commit_sha]

    # Create an array to store the commit shas
    commit_shas = []

    # Loop n times to generate n random commit shas
    for i in range(n):
        # Generate a random index within the range of possible commit SHAs
        random_index = random.randint(0, len(possible_commit_shas) - 1)

        # Get the commit SHA at the random index
        commit_sha = possible_commit_shas[random_index]

        # Append the commit SHA to the array
        commit_shas.append(commit_sha)

        # Remove the chosen commit SHA from the list of possible commit SHAs to avoid duplicates
        possible_commit_shas.remove(commit_sha)

    # Return the array of commit shas
    return commit_shas
