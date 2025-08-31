import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

def test_student_tracks_teacher():
    X, y = make_classification(n_samples=200, n_features=5, random_state=0)
    teacher = RandomForestClassifier(n_estimators=50, random_state=0)
    teacher.fit(X, y)
    teacher_probs = teacher.predict_proba(X)[:, 1]

    X_rep = np.vstack([X, X])
    y_rep = np.concatenate([np.ones(len(X)), np.zeros(len(X))])
    sample_weight = np.concatenate([teacher_probs, 1 - teacher_probs])

    student = LogisticRegression(max_iter=200)
    student.fit(X_rep, y_rep, sample_weight=sample_weight)
    student_probs = student.predict_proba(X)[:, 1]

    diff = np.abs(student_probs - teacher_probs).mean()
    assert diff < 0.1
